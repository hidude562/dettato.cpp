// The grouper's decode (dettato/grouper/decode.py), in double precision on the CPU.
//
// It reproduces the Python reference bit for bit where the reference is deterministic, which takes some care:
// - scipy's average linkage (nn_chain + stable sort + label), fcluster(..., "distance") and linear_sum_assignment are
//   ported line by line, so ties break the same way;
// - numpy's float32 / float64 matrix products are evaluated in the order OpenBLAS's Haswell kernels use (numpy 2.4 wheels
//   bundle OpenBLAS 0.3.31): GEMM / SYRK as one fused multiply-add chain per element, GEMV_T and DOT with their lane
//   splits; sums, means and norms with numpy's pairwise summation;
// - np.argsort's default (quick)sort on AVX2 machines is Intel's x86-simd-sort, which is not stable: its 64-bit argsort
//   (bitonic networks <= 256 elements, vectorised quicksort above) is emulated lane by lane, so equal track sizes come out
//   in the same order;
// - scipy.special.erf is Cephes' erf, not the C library's.
// Floating-point contraction is disabled for this file: every fused multiply-add below is an explicit std::fma.

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

#include "grouper/decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <type_traits>

namespace dtt
{

namespace
{
    constexpr double INF = std::numeric_limits<double>::infinity();

    // ------------------------------------------------------------------------------------------- numpy reductions
    /** numpy's pairwise summation (loops_utils.h.src) of n contiguous values, accumulating in T. */
    template <typename T>
    T pairwiseSum(const T* a, int64_t n)
    {
        if (n < 8) {
            T res = 0;
            for (int64_t i = 0; i < n; i++) {
                res = res + a[i];
            }
            return res;
        }
        if (n <= 128) {
            T r[8];
            for (int j = 0; j < 8; j++) {
                r[j] = a[j];
            }
            int64_t i = 8;
            for (; i < n - (n % 8); i += 8) {
                for (int j = 0; j < 8; j++) {
                    r[j] = r[j] + a[i + j];
                }
            }
            T res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++) {
                res = res + a[i];
            }
            return res;
        }
        int64_t n2 = n / 2;
        n2 -= n2 % 8;
        return pairwiseSum(a, n2) + pairwiseSum(a + n2, n - n2);
    }

    /** np.add.reduce over a whole contiguous 1-D array: the identity plus the pairwise sum. */
    template <typename T>
    T npSum(const T* a, int64_t n)
    {
        return T(0) + pairwiseSum(a, n);
    }

    /** np.linalg.norm(x) of one contiguous row: sqrt of the pairwise sum of the squares. */
    template <typename T>
    T npNorm(const T* a, int64_t n)
    {
        std::vector<T> sq(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; i++) {
            sq[i] = a[i] * a[i];
        }
        return std::sqrt(npSum(sq.data(), n));
    }

    /** x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9), rows of length K, in place (double). */
    void normaliseRows(std::vector<double>& x, int64_t rows, int64_t K)
    {
        for (int64_t r = 0; r < rows; r++) {
            double* row = x.data() + r * K;
            const double nn = std::max(npNorm(row, K), 1e-9);
            for (int64_t k = 0; k < K; k++) {
                row[k] = row[k] / nn;
            }
        }
    }

    // ------------------------------------------------------------------------------------------- numpy matmul
    // numpy 2.4 calls cblas_?syrk for A @ A.T, cblas_?gemm for other matrix products, cblas_?gemv for matrix @ vector and
    // cblas_?dot when the result is 1 x 1. The orders below are those of OpenBLAS 0.3.31's Haswell kernels.

    /** One element of DGEMM: a single fused multiply-add chain over k. */
    double ddotChain(const double* a, const double* b, int K)
    {
        double s = 0.0;
        for (int k = 0; k < K; k++) {
            s = std::fma(a[k], b[k], s);
        }
        return s;
    }

    /**
     * All pairs of rows (j >= i) of a (m, K): G[i, j] = G[j, i] = one FMA chain over k, as SSYRK / DSYRK compute each
     * element. aT is a transposed (K, m) copy; the k-outer loop keeps the per-element order and lets the j loop vectorise.
     */
    template <typename T>
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline void gramImpl(const T* aT, int m, int K, T* G)
    {
        std::vector<T> acc(static_cast<size_t>(m));
        for (int i = 0; i < m; i++) {
            T* ac = acc.data();
            for (int j = i; j < m; j++) {
                ac[j] = T(0);
            }
            for (int k = 0; k < K; k++) {
                const T a = aT[static_cast<size_t>(k) * m + i];
                const T* b = aT + static_cast<size_t>(k) * m;
                for (int j = i; j < m; j++) {
                    ac[j] = std::fma(a, b[j], ac[j]);
                }
            }
            for (int j = i; j < m; j++) {
                G[static_cast<size_t>(i) * m + j] = ac[j];
                G[static_cast<size_t>(j) * m + i] = ac[j];
            }
        }
    }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx2,fma"))) void gramF32Fma(const float* aT, int m, int K, float* G) { gramImpl(aT, m, K, G); }
    __attribute__((target("avx2,fma"))) void gramF64Fma(const double* aT, int m, int K, double* G) { gramImpl(aT, m, K, G); }
    bool cpuFma()
    {
        static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        return ok;
    }
#endif

    /** a (m, K) row-major -> G (m, m), exactly as numpy's a @ a.T (SYRK). */
    template <typename T>
    std::vector<T> gramRows(const T* a, int m, int K)
    {
        std::vector<T> aT(static_cast<size_t>(m) * K), G(static_cast<size_t>(m) * m);
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < K; k++) {
                aT[static_cast<size_t>(k) * m + i] = a[static_cast<size_t>(i) * K + k];
            }
        }
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        if (cpuFma()) {
            if constexpr (std::is_same_v<T, float>) {
                gramF32Fma(aT.data(), m, K, G.data());
            } else {
                gramF64Fma(aT.data(), m, K, G.data());
            }
            return G;
        }
#endif
        gramImpl(aT.data(), m, K, G.data());
        return G;
    }

    /** DGEMM's 4x1 tile (dgemm_kernel_4x8_haswell.S KERNEL4x1): four accumulators over k mod 4 in blocks of 8. */
    double ddotTile4x1(const double* a, const double* b, int K)
    {
        double acc[4] = {0.0, 0.0, 0.0, 0.0};
        const int K8 = K & ~7;
        for (int k = 0; k < K8; k++) {
            acc[k & 3] = std::fma(a[k], b[k], acc[k & 3]);
        }
        for (int k = K8; k < K; k++) {
            acc[0] = std::fma(a[k], b[k], acc[0]);
        }
        return (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }

    /** cblas_ddot (ddot.c + ddot_microk_haswell-2.c), as numpy's DOUBLE_dot uses it (0. + ddot). */
    double ddotBlas(const double* x, const double* y, int n)
    {
        const int n1 = n & -16;
        double dot = 0.0;
        if (n1 != 0) {
            double R[4][4] = {};
            for (int k = 0; k < n1; k++) {
                R[(k >> 2) & 3][k & 3] = std::fma(x[k], y[k], R[(k >> 2) & 3][k & 3]);
            }
            double h[4][2];
            for (int q = 0; q < 4; q++) {
                h[q][0] = R[q][0] + R[q][2];
                h[q][1] = R[q][1] + R[q][3];
            }
            const double z0 = (h[0][0] + h[1][0]) + (h[2][0] + h[3][0]);
            const double z1 = (h[0][1] + h[1][1]) + (h[2][1] + h[3][1]);
            dot = z0 + z1;
        }
        for (int i = n1; i < n; i++) {
            dot = dot + y[i] * x[i];
        }
        return 0.0 + dot;
    }

    /**
     * One output of cblas_dgemv (Trans, column-major) = OpenBLAS dgemv_t_4.c on Haswell: y[j] = A[j, :] . x for a matrix of
     * r rows of length K. The kernel used (and so the summation order) depends on the row's place among the r rows.
     */
#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline double gemvRowImpl(const double* A, int r, int K, const double* x, int j)
    {
        const int m3 = K & 3, NB = K - m3;
        const double* a = A + static_cast<size_t>(j) * K;
        double y = 0.0;
        if (NB > 0) {
            const int n4 = 4 * (r >> 2);
            if (j < n4) { // dgemv_kernel_4x4: 4 FMA lanes, (l0 + l2) + (l1 + l3)
                double acc[4] = {0.0, 0.0, 0.0, 0.0};
                for (int k = 0; k < NB; k++) {
                    acc[k & 3] = std::fma(a[k], x[k], acc[k & 3]);
                }
                y = y + ((acc[0] + acc[2]) + (acc[1] + acc[3]));
            } else if ((r & 2) && j < n4 + 2) { // dgemv_kernel_4x2: SSE, 2 lanes, multiply then add
                double l[2] = {0.0, 0.0};
                for (int k = 0; k < NB; k++) {
                    l[k & 1] = l[k & 1] + a[k] * x[k];
                }
                y = y + (l[0] + l[1]);
            } else { // dgemv_kernel_4x1: two SSE accumulators over k mod 4
                double l[4] = {0.0, 0.0, 0.0, 0.0};
                for (int k = 0; k < NB; k++) {
                    l[k & 3] = l[k & 3] + a[k] * x[k];
                }
                y = y + ((l[0] + l[2]) + (l[1] + l[3]));
            }
        }
        if (m3 == 1) { // the K % 4 tail, C code compiled with contraction
            y = std::fma(a[NB], x[NB], y);
        } else if (m3 > 1) {
            double e = std::fma(a[NB], x[NB], a[NB + 1] * x[NB + 1]);
            if (m3 == 3) {
                e = std::fma(a[NB + 2], x[NB + 2], e);
            }
            y = y + e;
        }
        return y;
    }

#if defined(__GNUC__) || defined(__clang__)
    __attribute__((always_inline))
#endif
    inline void gemvRowsImpl(const double* A, int r, int K, const double* x, double* y, const char* rows)
    {
        for (int j = 0; j < r; j++) {
            if (rows == nullptr || rows[j]) {
                y[j] = gemvRowImpl(A, r, K, x, j);
            }
        }
    }

#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    __attribute__((target("avx2,fma"))) void gemvRowsFma(const double* A, int r, int K, const double* x, double* y, const char* rows)
    {
        gemvRowsImpl(A, r, K, x, y, rows);
    }
#endif

    /** cblas_dgemv for the rows with rows[j] != 0 (all rows when rows is null); the other outputs are left as they are. */
    void dgemvT(const double* A, int r, int K, const double* x, double* y, const char* rows = nullptr)
    {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
        if (cpuFma()) {
            gemvRowsFma(A, r, K, x, y, rows);
            return;
        }
#endif
        gemvRowsImpl(A, r, K, x, y, rows);
    }

    /** numpy: A (r, K) @ x (K,) -> (r,). */
    std::vector<double> matVec(const double* A, int r, int K, const double* x)
    {
        std::vector<double> y(static_cast<size_t>(r));
        if (r == 1) {
            y[0] = ddotBlas(A, x, K);
        } else if (r > 1) {
            dgemvT(A, r, K, x, y.data());
        }
        return y;
    }

    /** numpy: A (n, K) @ B.T with B (p, K) -> (n, p) row-major. */
    std::vector<double> matMulT(const double* A, int n, int K, const double* B, int p)
    {
        std::vector<double> C(static_cast<size_t>(n) * p);
        if (n == 0 || p == 0) {
            return C;
        }
        if (n == 1 && p == 1) {
            C[0] = ddotBlas(A, B, K);
        } else if (n == 1) { // vector @ matrix: gemv over the rows of B
            dgemvT(B, p, K, A, C.data());
        } else if (p == 1) { // matrix @ vector
            dgemvT(A, n, K, B, C.data());
        } else {
            // DGEMM, column-major C^T = B A^T: OpenBLAS's M = p, N = n. Every tile is one FMA chain per element except
            // the N = 1 remainder tile (numpy's last row when n is odd) with M = 4 columns, which splits k over four
            // accumulators. (Single-threaded layout; OpenBLAS splits large products over threads.)
            const int p4 = p & ~3;
            for (int i = 0; i < n; i++) {
                const bool tile1 = (n & 1) && i == n - 1;
                for (int j = 0; j < p; j++) {
                    const double* a = A + static_cast<size_t>(i) * K;
                    const double* b = B + static_cast<size_t>(j) * K;
                    C[static_cast<size_t>(i) * p + j] = (tile1 && j < p4) ? ddotTile4x1(a, b, K) : ddotChain(a, b, K);
                }
            }
        }
        return C;
    }


    template <typename T>
    int64_t argmaxFirst(const T* v, int64_t n)
    {
        int64_t k = 0;
        for (int64_t i = 1; i < n; i++) {
            if (v[i] > v[k]) {
                k = i;
            }
        }
        return k;
    }

    // ------------------------------------------------------------------------------------------- np.argsort (AVX2)
    // x86-simd-sort (numpy 2.4.6's vendored copy, 5adb334) avx2_argsort<double>: 4-lane key/index registers.
    struct KV {
        double k[4];
        uint64_t i[4];
    };
    constexpr int SWAP2[4] = {1, 0, 3, 2};
    constexpr int SWAP4[4] = {2, 3, 0, 1};
    constexpr int REV4[4] = {3, 2, 1, 0};

    KV permute(const KV& v, const int* p)
    {
        KV o;
        for (int l = 0; l < 4; l++) {
            o.k[l] = v.k[p[l]];
            o.i[l] = v.i[p[l]];
        }
        return o;
    }

    inline double vmin(double a, double b) { return a < b ? a : b; } // _mm256_min_pd(a, b)
    inline double vmax(double a, double b) { return a > b ? a : b; } // _mm256_max_pd(a, b)

    /** cmp_merge with indices: lanes in mask take the max, the others the min; an index follows its key. */
    void cmpMerge(KV& v, const int* perm, unsigned mask)
    {
        const KV o = permute(v, perm);
        KV r;
        for (int l = 0; l < 4; l++) {
            const double mn = vmin(o.k[l], v.k[l]), mx = vmax(o.k[l], v.k[l]);
            r.k[l] = ((mask >> l) & 1) ? mx : mn;
            r.i[l] = r.k[l] == v.k[l] ? v.i[l] : o.i[l];
        }
        v = r;
    }

    void sortReg4(KV& v)
    {
        cmpMerge(v, SWAP2, 0xA);
        cmpMerge(v, REV4, 0xC);
        cmpMerge(v, SWAP2, 0xA);
    }

    void bitonicMergeReg4(KV& v)
    {
        cmpMerge(v, SWAP4, 0xC);
        cmpMerge(v, SWAP2, 0xA);
    }

    void coex(KV& a, KV& b)
    {
        for (int l = 0; l < 4; l++) {
            const double k1 = vmin(a.k[l], b.k[l]), k2 = vmax(a.k[l], b.k[l]);
            const bool eq = k1 == a.k[l];
            const uint64_t i1 = eq ? a.i[l] : b.i[l], i2 = eq ? b.i[l] : a.i[l];
            a.k[l] = k1;
            b.k[l] = k2;
            a.i[l] = i1;
            b.i[l] = i2;
        }
    }

    void bitonicMergeN(KV* v, int numVecs)
    {
        for (int i = 0; i < numVecs / 2; i++) {
            KV& hi = v[numVecs - i - 1];
            hi = permute(hi, REV4);
            coex(v[i], hi);
            hi = permute(hi, REV4);
        }
        for (int num = numVecs / 2; num >= 2; num /= 2) {
            for (int j = 0; j < numVecs; j += num) {
                for (int i = 0; i < num / 2; i++) {
                    coex(v[i + j], v[i + j + num / 2]);
                }
            }
        }
        for (int i = 0; i < numVecs; i++) {
            bitonicMergeReg4(v[i]);
        }
    }

    /** argsort_n<.., 256>: bitonic network over up to 256 elements, arg[0..N) indexes arr. */
    void argsortN(const double* arr, uint64_t* arg, int N)
    {
        int numVecs = 64;
        while (numVecs > 1 && N * 2 <= numVecs * 4) {
            numVecs /= 2;
        }
        std::vector<KV> v(static_cast<size_t>(numVecs));
        for (int i = 0; i < numVecs; i++) {
            for (int l = 0; l < 4; l++) {
                const int p = 4 * i + l;
                if (i < numVecs / 2 || p < N) {
                    v[i].i[l] = arg[p];
                    v[i].k[l] = arr[arg[p]];
                } else {
                    v[i].i[l] = std::numeric_limits<uint64_t>::max();
                    v[i].k[l] = INF;
                }
            }
        }
        for (auto& r : v) {
            sortReg4(r);
        }
        for (int per = 2; per <= numVecs; per *= 2) {
            for (int i = 0; i < numVecs / per; i++) {
                bitonicMergeN(v.data() + i * per, per);
            }
        }
        for (int p = 0; p < N; p++) {
            arg[p] = v[p / 4].i[p % 4];
        }
    }

    /** partition_vec_avx2: lanes >= pivot to the right store (in lane order from the top), the rest to the left. */
    int partitionVec(uint64_t* arg, int64_t left, int64_t right, const KV& v, double pivot, double* minv, double* maxv)
    {
        uint64_t temp[4];
        int lo = 0, hi = 3, ge = 0;
        for (int l = 0; l < 4; l++) {
            if (v.k[l] >= pivot) {
                temp[hi--] = v.i[l];
                ge++;
            } else {
                temp[lo++] = v.i[l];
            }
        }
        for (int l = 0; l < 4; l++) {
            arg[left + l] = temp[l];
        }
        for (int l = 0; l < 4; l++) {
            arg[right - 4 + l] = temp[l];
        }
        for (int l = 0; l < 4; l++) {
            minv[l] = vmin(v.k[l], minv[l]);
            maxv[l] = vmax(v.k[l], maxv[l]);
        }
        return ge;
    }

    KV gather(const double* arr, const uint64_t* arg)
    {
        KV v;
        for (int l = 0; l < 4; l++) {
            v.i[l] = arg[l];
            v.k[l] = arr[arg[l]];
        }
        return v;
    }

    double reduceMin(const double* m)
    {
        double t[4];
        for (int l = 0; l < 4; l++) {
            t[l] = vmin(m[l], m[SWAP2[l]]);
        }
        return t[3] < t[0] ? t[3] : t[0];
    }

    double reduceMax(const double* m)
    {
        double t[4];
        for (int l = 0; l < 4; l++) {
            t[l] = vmax(m[l], m[SWAP2[l]]);
        }
        return t[0] < t[3] ? t[3] : t[0];
    }

    /** One scalar step of the partitions' prologue. */
    void scalarStep(const double* arr, uint64_t* arg, int64_t& left, int64_t& right, double pivot, double& smallest,
                    double& biggest)
    {
        const double x = arr[arg[left]];
        smallest = x < smallest ? x : smallest; // std::min(smallest, x, a < b)
        biggest = biggest < x ? x : biggest;    // std::max(biggest, x, a < b)
        if (!(x < pivot)) {
            std::swap(arg[left], arg[--right]);
        } else {
            ++left;
        }
    }

    int64_t argPartition(const double* arr, uint64_t* arg, int64_t left, int64_t right, double pivot, double& smallest,
                         double& biggest)
    {
        for (int64_t i = (right - left) % 4; i > 0; --i) {
            scalarStep(arr, arg, left, right, pivot, smallest, biggest);
        }
        if (left == right) {
            return left;
        }
        double minv[4], maxv[4];
        std::fill(minv, minv + 4, smallest);
        std::fill(maxv, maxv + 4, biggest);
        if (right - left == 4) {
            const KV v = gather(arr, arg + left);
            const int ge = partitionVec(arg, left, left + 4, v, pivot, minv, maxv);
            smallest = reduceMin(minv);
            biggest = reduceMax(maxv);
            return left + (4 - ge);
        }
        const KV vl = gather(arr, arg + left), vr = gather(arr, arg + right - 4);
        int64_t r_store = right - 4, l_store = left;
        left += 4;
        right -= 4;
        while (right - left != 0) {
            KV cur;
            if ((r_store + 4) - right < left - l_store) {
                right -= 4;
                cur = gather(arr, arg + right);
            } else {
                cur = gather(arr, arg + left);
                left += 4;
            }
            const int ge = partitionVec(arg, l_store, r_store + 4, cur, pivot, minv, maxv);
            r_store -= ge;
            l_store += 4 - ge;
        }
        int ge = partitionVec(arg, l_store, r_store + 4, vl, pivot, minv, maxv);
        l_store += 4 - ge;
        ge = partitionVec(arg, l_store, l_store + 4, vr, pivot, minv, maxv);
        l_store += 4 - ge;
        smallest = reduceMin(minv);
        biggest = reduceMax(maxv);
        return l_store;
    }

    int64_t argPartitionUnrolled(const double* arr, uint64_t* arg, int64_t left, int64_t right, double pivot,
                                 double& smallest, double& biggest)
    {
        constexpr int U = 4;
        if (right - left <= 8 * U * 4) {
            return argPartition(arr, arg, left, right, pivot, smallest, biggest);
        }
        for (int64_t i = (right - left) % (U * 4); i > 0; --i) {
            scalarStep(arr, arg, left, right, pivot, smallest, biggest);
        }
        if (left == right) {
            return left;
        }
        double minv[4], maxv[4];
        std::fill(minv, minv + 4, smallest);
        std::fill(maxv, maxv + 4, biggest);
        KV vl[U], vr[U];
        for (int ii = 0; ii < U; ii++) {
            vl[ii] = gather(arr, arg + left + 4 * ii);
            vr[ii] = gather(arr, arg + (right - 4 * (U - ii)));
        }
        int64_t r_store = right - 4, l_store = left;
        left += U * 4;
        right -= U * 4;
        while (right - left != 0) {
            KV cur[U];
            if ((r_store + 4) - right < left - l_store) {
                right -= U * 4;
                for (int ii = 0; ii < U; ii++) {
                    cur[ii] = gather(arr, arg + right + ii * 4);
                }
            } else {
                for (int ii = 0; ii < U; ii++) {
                    cur[ii] = gather(arr, arg + left + ii * 4);
                }
                left += U * 4;
            }
            for (int ii = 0; ii < U; ii++) {
                const int ge = partitionVec(arg, l_store, r_store + 4, cur[ii], pivot, minv, maxv);
                l_store += 4 - ge;
                r_store -= ge;
            }
        }
        for (int ii = 0; ii < U; ii++) {
            const int ge = partitionVec(arg, l_store, r_store + 4, vl[ii], pivot, minv, maxv);
            l_store += 4 - ge;
            r_store -= ge;
        }
        for (int ii = 0; ii < U; ii++) {
            const int ge = partitionVec(arg, l_store, r_store + 4, vr[ii], pivot, minv, maxv);
            l_store += 4 - ge;
            r_store -= ge;
        }
        smallest = reduceMin(minv);
        biggest = reduceMax(maxv);
        return l_store;
    }

    void argsortRec(const double* arr, uint64_t* arg, int64_t left, int64_t right, int64_t max_iters)
    {
        if (max_iters <= 0) {
            std::sort(arg + left, arg + right + 1, [arr](uint64_t a, uint64_t b) { return arr[a] < arr[b]; });
            return;
        }
        if (right + 1 - left <= 256) {
            argsortN(arr, arg + left, static_cast<int>(right + 1 - left));
            return;
        }
        double pivot;
        if (right - left >= 4) {
            const int64_t size = (right - left) / 4;
            double s[4] = {arr[arg[left + size]], arr[arg[left + 2 * size]], arr[arg[left + 3 * size]], arr[arg[left + 4 * size]]};
            std::sort(s, s + 4);
            pivot = s[2];
        } else {
            pivot = arr[arg[right]];
        }
        double smallest = INF, biggest = -INF;
        const int64_t pi = argPartitionUnrolled(arr, arg, left, right + 1, pivot, smallest, biggest);
        if (pivot != smallest) {
            argsortRec(arr, arg, left, pi - 1, max_iters - 1);
        }
        if (pivot != biggest) {
            argsortRec(arr, arg, pi, right, max_iters - 1);
        }
    }
} // namespace

/** np.argsort(a) (kind "quicksort") as numpy 2.4 runs it on an AVX2 x86-64 CPU (x86-simd-sort), for NaN-free float64. */
std::vector<int64_t> npArgsort(const std::vector<double>& a)
{
    const int64_t n = static_cast<int64_t>(a.size());
    std::vector<uint64_t> arg(static_cast<size_t>(n));
    std::iota(arg.begin(), arg.end(), uint64_t{0});
    if (n > 1 && !std::is_sorted(a.begin(), a.end())) {
        argsortRec(a.data(), arg.data(), 0, n - 1, 2 * static_cast<int64_t>(std::log2(static_cast<double>(n))));
    }
    return std::vector<int64_t>(arg.begin(), arg.end());
}

namespace
{
    // ------------------------------------------------------------------------------------------- scipy.special.erf
    // Cephes ndtr.h (xsf f7b85f5, scipy 1.18.0).
    double polevl(double x, const double* coef, int N)
    {
        double ans = *coef++;
        int i = N;
        do {
            ans = ans * x + *coef++;
        } while (--i);
        return ans;
    }

    double p1evl(double x, const double* coef, int N)
    {
        double ans = x + *coef++;
        int i = N - 1;
        do {
            ans = ans * x + *coef++;
        } while (--i);
        return ans;
    }

    constexpr double NDTR_P[] = {2.46196981473530512524E-10, 5.64189564831068821977E-1, 7.46321056442269912687E0,
                                 4.86371970985681366614E1,   1.96520832956077098242E2,  5.26445194995477358631E2,
                                 9.34528527171957607540E2,   1.02755188689515710272E3,  5.57535335369399327526E2};
    constexpr double NDTR_Q[] = {1.32281951154744992508E1, 8.67072140885989742329E1, 3.54937778887819891062E2,
                                 9.75708501743205489753E2, 1.82390916687909736289E3, 2.24633760818710981792E3,
                                 1.65666309194161350182E3, 5.57535340817727675546E2};
    constexpr double NDTR_R[] = {5.64189583547755073984E-1, 1.27536670759978104416E0, 5.01905042251180477414E0,
                                 6.16021097993053585195E0,  7.40974269950448939160E0, 2.97886665372100240670E0};
    constexpr double NDTR_S[] = {2.26052863220117276590E0, 9.39603524938001434673E0, 1.20489539808096656605E1,
                                 1.70814450747565897222E1, 9.60896809063285878198E0, 3.36907645100081516050E0};
    constexpr double NDTR_T[] = {9.60497373987051638749E0, 9.00260197203842689217E1, 2.23200534594684319226E3,
                                 7.00332514112805075473E3, 5.55923013010394962768E4};
    constexpr double NDTR_U[] = {3.35617141647503099647E1, 5.21357949780152679795E2, 4.59432382970980127987E3,
                                 2.26290000613890934246E4, 4.92673942608635921086E4};
    constexpr double MAXLOG = 7.097827128933839730962063185871E2;

    double cephesErf(double x);

    double cephesErfc(double a)
    {
        if (std::isnan(a)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double x = a < 0.0 ? -a : a;
        if (x < 1.0) {
            return 1.0 - cephesErf(a);
        }
        double z = -a * a;
        if (z < -MAXLOG) {
            return a < 0 ? 2.0 : 0.0;
        }
        z = std::exp(z);
        double p, q;
        if (x < 8.0) {
            p = polevl(x, NDTR_P, 8);
            q = p1evl(x, NDTR_Q, 8);
        } else {
            p = polevl(x, NDTR_R, 5);
            q = p1evl(x, NDTR_S, 6);
        }
        double y = (z * p) / q;
        if (a < 0) {
            y = 2.0 - y;
        }
        if (y != 0.0) {
            return y;
        }
        return a < 0 ? 2.0 : 0.0;
    }

    double cephesErf(double x)
    {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < 0.0) {
            return -cephesErf(-x);
        }
        if (std::abs(x) > 1.0) {
            return 1.0 - cephesErfc(x);
        }
        const double z = x * x;
        return x * polevl(z, NDTR_T, 4) / p1evl(z, NDTR_U, 5);
    }
} // namespace

double scipyErf(double x)
{
    return cephesErf(x);
}

// ----------------------------------------------------------------------------------------------- scipy hierarchy
namespace
{
    inline int64_t condensedIndex(int64_t n, int64_t i, int64_t j)
    {
        if (i < j) {
            return n * i - (i * (i + 1) / 2) + (j - i - 1);
        }
        return n * j - (j * (j + 1) / 2) + (i - j - 1);
    }

    struct ZRow {
        double a, b, d, s;
    };

    /** _hierarchy.nn_chain(dists, n, method=average), then the stable sort by distance and label(). */
    std::vector<ZRow> linkageAverage(const std::vector<double>& dists, int n)
    {
        std::vector<ZRow> Z(static_cast<size_t>(n - 1));
        std::vector<double> D = dists;
        std::vector<int> size(n, 1), chain(n);
        int chain_length = 0, y = 0;
        for (int k = 0; k < n - 1; k++) {
            if (chain_length == 0) {
                chain_length = 1;
                for (int i = 0; i < n; i++) {
                    if (size[i] > 0) {
                        chain[0] = i;
                        break;
                    }
                }
            }
            int x;
            double current_min;
            while (true) {
                x = chain[chain_length - 1];
                if (chain_length > 1) { // prefer the previous element of the chain on ties
                    y = chain[chain_length - 2];
                    current_min = D[condensedIndex(n, x, y)];
                } else {
                    current_min = INF;
                }
                for (int i = 0; i < n; i++) {
                    if (size[i] == 0 || x == i) {
                        continue;
                    }
                    const double dist = D[condensedIndex(n, x, i)];
                    if (dist < current_min) {
                        current_min = dist;
                        y = i;
                    }
                }
                if (chain_length > 1 && y == chain[chain_length - 2]) {
                    break;
                }
                chain[chain_length] = y;
                chain_length++;
            }
            chain_length -= 2;
            if (x > y) {
                std::swap(x, y);
            }
            const int nx = size[x], ny = size[y];
            Z[k] = {static_cast<double>(x), static_cast<double>(y), current_min, static_cast<double>(nx + ny)};
            size[x] = 0;
            size[y] = nx + ny;
            for (int i = 0; i < n; i++) {
                const int ni = size[i];
                if (ni == 0 || i == y) {
                    continue;
                }
                const double d_xi = D[condensedIndex(n, i, x)], d_yi = D[condensedIndex(n, i, y)];
                D[condensedIndex(n, i, y)] = (nx * d_xi + ny * d_yi) / (nx + ny);
            }
        }
        std::stable_sort(Z.begin(), Z.end(), [](const ZRow& p, const ZRow& q) { return p.d < q.d; });
        // label(): union-find over the sorted merges
        std::vector<int> parent(2 * n - 1), usz(2 * n - 1, 1);
        std::iota(parent.begin(), parent.end(), 0);
        int next = n;
        auto find = [&](int v) {
            int r = v;
            while (parent[r] != r) {
                r = parent[r];
            }
            while (parent[v] != r) {
                const int nv = parent[v];
                parent[v] = r;
                v = nv;
            }
            return r;
        };
        for (int i = 0; i < n - 1; i++) {
            const int xr = find(static_cast<int>(Z[i].a)), yr = find(static_cast<int>(Z[i].b));
            Z[i].a = std::min(xr, yr);
            Z[i].b = std::max(xr, yr);
            parent[xr] = next;
            parent[yr] = next;
            usz[next] = usz[xr] + usz[yr];
            Z[i].s = usz[next];
            next++;
        }
        return Z;
    }

    /** _hierarchy.cluster_dist: get_max_dist_for_each_cluster + cluster_monocrit -> T (1-based). */
    std::vector<int> clusterDist(const std::vector<ZRow>& Z, double cutoff, int n)
    {
        std::vector<double> MD(static_cast<size_t>(n));
        std::vector<int> curr(static_cast<size_t>(n));
        std::vector<char> visited(static_cast<size_t>(2 * n), 0);
        int k = 0;
        curr[0] = 2 * n - 2;
        while (k >= 0) {
            const int root = curr[k] - n;
            const int lc = static_cast<int>(Z[root].a), rc = static_cast<int>(Z[root].b);
            if (lc >= n && !visited[lc]) {
                visited[lc] = 1;
                curr[++k] = lc;
                continue;
            }
            if (rc >= n && !visited[rc]) {
                visited[rc] = 1;
                curr[++k] = rc;
                continue;
            }
            double max_dist = Z[root].d;
            if (lc >= n && MD[lc - n] > max_dist) {
                max_dist = MD[lc - n];
            }
            if (rc >= n && MD[rc - n] > max_dist) {
                max_dist = MD[rc - n];
            }
            MD[root] = max_dist;
            k--;
        }
        std::vector<int> T(static_cast<size_t>(n), 0);
        std::fill(visited.begin(), visited.end(), 0);
        int n_cluster = 0, leader = -1;
        k = 0;
        curr[0] = 2 * n - 2;
        while (k >= 0) {
            const int root = curr[k] - n;
            const int lc = static_cast<int>(Z[root].a), rc = static_cast<int>(Z[root].b);
            if (leader == -1 && MD[root] <= cutoff) {
                leader = root;
                n_cluster++;
            }
            if (lc >= n && !visited[lc]) {
                visited[lc] = 1;
                curr[++k] = lc;
                continue;
            }
            if (rc >= n && !visited[rc]) {
                visited[rc] = 1;
                curr[++k] = rc;
                continue;
            }
            if (lc < n) {
                if (leader == -1) {
                    n_cluster++;
                }
                T[lc] = n_cluster;
            }
            if (rc < n) {
                if (leader == -1) {
                    n_cluster++;
                }
                T[rc] = n_cluster;
            }
            if (leader == root) {
                leader = -1;
            }
            k--;
        }
        return T;
    }
} // namespace

std::vector<int> fclusterAverage(const std::vector<double>& y, int m, double t)
{
    if (m <= 0) {
        return {};
    }
    if (m == 1) {
        return {0};
    }
    if (static_cast<int64_t>(y.size()) != static_cast<int64_t>(m) * (m - 1) / 2) {
        throw std::invalid_argument("fclusterAverage: condensed matrix size mismatch");
    }
    std::vector<int> T = clusterDist(linkageAverage(y, m), t, m);
    for (int& v : T) {
        v -= 1;
    }
    return T;
}

// ----------------------------------------------------------------------------------------------- scipy LSAP
namespace
{
    int64_t augmentingPath(int64_t nc, const double* cost, std::vector<double>& u, std::vector<double>& v,
                           std::vector<int64_t>& path, std::vector<int64_t>& row4col, std::vector<double>& spc, int64_t i,
                           std::vector<char>& SR, std::vector<char>& SC, std::vector<int64_t>& remaining, double* p_minVal)
    {
        double minVal = 0;
        int64_t num_remaining = nc;
        for (int64_t it = 0; it < nc; it++) {
            remaining[it] = nc - it - 1;
        }
        std::fill(SR.begin(), SR.end(), 0);
        std::fill(SC.begin(), SC.end(), 0);
        std::fill(spc.begin(), spc.end(), INF);
        int64_t sink = -1;
        while (sink == -1) {
            int64_t index = -1;
            double lowest = INF;
            SR[i] = 1;
            for (int64_t it = 0; it < num_remaining; it++) {
                const int64_t j = remaining[it];
                const double r = minVal + cost[i * nc + j] - u[i] - v[j];
                if (r < spc[j]) {
                    path[j] = i;
                    spc[j] = r;
                }
                if (spc[j] < lowest || (spc[j] == lowest && row4col[j] == -1)) {
                    lowest = spc[j];
                    index = it;
                }
            }
            minVal = lowest;
            if (minVal == INF) {
                return -1;
            }
            const int64_t j = remaining[index];
            if (row4col[j] == -1) {
                sink = j;
            } else {
                i = row4col[j];
            }
            SC[j] = 1;
            remaining[index] = remaining[--num_remaining];
        }
        *p_minVal = minVal;
        return sink;
    }
} // namespace

void linearSumAssignment(const std::vector<double>& cost_in, int nr_in, int nc_in, std::vector<int>& rows, std::vector<int>& cols)
{
    rows.clear();
    cols.clear();
    int64_t nr = nr_in, nc = nc_in;
    if (nr == 0 || nc == 0) {
        return;
    }
    const bool transpose = nc < nr;
    std::vector<double> cost(cost_in.begin(), cost_in.begin() + nr * nc);
    if (transpose) {
        std::vector<double> t(static_cast<size_t>(nr * nc));
        for (int64_t i = 0; i < nr; i++) {
            for (int64_t j = 0; j < nc; j++) {
                t[j * nr + i] = cost[i * nc + j];
            }
        }
        cost.swap(t);
        std::swap(nr, nc);
    }
    for (double c : cost) {
        if (c != c || c == -INF) {
            throw std::invalid_argument("linearSumAssignment: invalid cost");
        }
    }
    std::vector<double> u(nr, 0), v(nc, 0), spc(nc);
    std::vector<int64_t> path(nc, -1), col4row(nr, -1), row4col(nc, -1), remaining(nc);
    std::vector<char> SR(nr), SC(nc);
    for (int64_t curRow = 0; curRow < nr; curRow++) {
        double minVal;
        const int64_t sink = augmentingPath(nc, cost.data(), u, v, path, row4col, spc, curRow, SR, SC, remaining, &minVal);
        if (sink < 0) {
            throw std::invalid_argument("linearSumAssignment: infeasible cost matrix");
        }
        u[curRow] += minVal;
        for (int64_t i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - spc[col4row[i]];
            }
        }
        for (int64_t j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - spc[j];
            }
        }
        int64_t j = sink;
        while (true) {
            const int64_t i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }
    if (transpose) {
        std::vector<int64_t> order(nr);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return col4row[a] < col4row[b]; });
        for (int64_t q : order) {
            rows.push_back(static_cast<int>(col4row[q]));
            cols.push_back(static_cast<int>(q));
        }
    } else {
        for (int64_t i = 0; i < nr; i++) {
            rows.push_back(static_cast<int>(i));
            cols.push_back(static_cast<int>(col4row[i]));
        }
    }
}

// ----------------------------------------------------------------------------------------------- decode.py
namespace
{
    /** np.unique(x, return_inverse=True). */
    void unique(const std::vector<int64_t>& x, std::vector<int64_t>& u, std::vector<int64_t>& inv)
    {
        u = x;
        std::sort(u.begin(), u.end());
        u.erase(std::unique(u.begin(), u.end()), u.end());
        inv.resize(x.size());
        for (size_t i = 0; i < x.size(); i++) {
            inv[i] = std::lower_bound(u.begin(), u.end(), x[i]) - u.begin();
        }
    }

    std::vector<int64_t> bincount(const std::vector<int64_t>& x, int64_t minlength)
    {
        int64_t n = minlength;
        for (int64_t v : x) {
            n = std::max(n, v + 1);
        }
        std::vector<int64_t> c(static_cast<size_t>(n), 0);
        for (int64_t v : x) {
            c[v]++;
        }
        return c;
    }

} // namespace

std::vector<double> groupWindowDistances(const GroupWindow& w, int E, int K, double lam)
{
        const int m = static_cast<int>(w.sel.size());
        // Dm = 1 - e @ e.T (float32 SSYRK)
        std::vector<float> Dm = gramRows(w.emb.data(), m, E);
        for (float& v : Dm) {
            v = 1.0f - v;
        }
        if (lam > 0) {
            std::vector<float> pn(w.probs.begin(), w.probs.end());
            for (int i = 0; i < m; i++) {
                float* row = &pn[static_cast<size_t>(i) * K];
                const float nn = std::max(npNorm(row, K), 1e-9f);
                for (int k = 0; k < K; k++) {
                    row[k] = row[k] / nn;
                }
            }
            const std::vector<float> Gp = gramRows(pn.data(), m, K);
            const float lamf = static_cast<float>(lam);
            for (size_t q = 0; q < Dm.size(); q++) {
                Dm[q] = Dm[q] + lamf * (1.0f - Gp[q]);
            }
        }
        // np.clip((Dm + Dm.T) / 2, 0, None) on a symmetric matrix, diagonal 0, squareform -> float64
        std::vector<double> y;
        y.reserve(static_cast<size_t>(m) * (m - 1) / 2);
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < m; j++) {
                const float a = Dm[static_cast<size_t>(i) * m + j];
                const float s = (a + a) / 2.0f;
                y.push_back(static_cast<double>(s >= 0.0f ? s : 0.0f));
            }
        }
        return y;
}

namespace
{
    /** _cluster: the float32 distance matrix exactly as numpy builds it, then scipy's average linkage cut at tau. */
    std::vector<int> clusterWindow(const GroupWindow& w, int E, int K, double tau, double lam)
    {
        const int m = static_cast<int>(w.sel.size());
        if (m == 1) {
            return {0};
        }
        return fclusterAverage(groupWindowDistances(w, E, K, lam), m, tau);
    }

    std::vector<int64_t> groupChain(const GroupPredictions& P, const std::vector<double>& on, double tau, double lam)
    {
        const int n = P.n;
        std::vector<int64_t> track(n, -1), owner(n, -1);
        std::vector<double> best_w(n, -1.0);
        int64_t n_glob = 0;
        const double W = P.W;
        std::vector<size_t> order(P.wins.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return P.wins[a].t0 < P.wins[b].t0; });
        for (size_t wi : order) {
            const GroupWindow& w = P.wins[wi];
            const int m = static_cast<int>(w.sel.size());
            if (m == 0) {
                continue;
            }
            const std::vector<int> lab = clusterWindow(w, P.E, P.K, tau, lam);
            const int nl = *std::max_element(lab.begin(), lab.end()) + 1;
            std::vector<int64_t> glob_of(nl, -1);
            std::vector<int64_t> prev(m), gids;
            for (int i = 0; i < m; i++) {
                prev[i] = owner[w.sel[i]];
                if (prev[i] >= 0) {
                    gids.push_back(prev[i]);
                }
            }
            if (!gids.empty()) {
                std::sort(gids.begin(), gids.end());
                gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
                const int ng = static_cast<int>(gids.size());
                std::vector<double> ov(static_cast<size_t>(nl) * ng, 0.0), cost(ov.size());
                for (int i = 0; i < m; i++) {
                    if (prev[i] >= 0) {
                        const int c = static_cast<int>(std::lower_bound(gids.begin(), gids.end(), prev[i]) - gids.begin());
                        ov[static_cast<size_t>(lab[i]) * ng + c] += 1.0;
                    }
                }
                for (size_t q = 0; q < ov.size(); q++) {
                    cost[q] = -ov[q];
                }
                std::vector<int> r, c;
                linearSumAssignment(cost, nl, ng, r, c);
                for (size_t q = 0; q < r.size(); q++) {
                    if (ov[static_cast<size_t>(r[q]) * ng + c[q]] > 0) {
                        glob_of[r[q]] = gids[c[q]];
                    }
                }
            }
            for (int li = 0; li < nl; li++) {
                if (glob_of[li] < 0) {
                    glob_of[li] = n_glob++;
                }
            }
            for (int i = 0; i < m; i++) {
                const int s = w.sel[i];
                const int64_t g = glob_of[lab[i]];
                owner[s] = g;
                const double wt = std::min(std::max(1.0 - std::abs((on[s] - w.t0) / W - 0.5) * 2, 0.05), 1.0);
                if (wt > best_w[s]) {
                    track[s] = g;
                    best_w[s] = wt;
                }
            }
        }
        return track;
    }

    /** _agglo([(me, 1.0), (mp, lam)], size, tau2). */
    std::vector<int64_t> agglo(std::vector<std::pair<std::vector<double>, double>> spaces, const std::vector<int>& dims,
                               std::vector<double> size, double tau2, int cap = 600)
    {
        std::vector<std::pair<std::vector<double>*, std::pair<int, double>>> sp;
        for (size_t s = 0; s < spaces.size(); s++) {
            if (spaces[s].second > 0) {
                sp.push_back({&spaces[s].first, {dims[s], spaces[s].second}});
            }
        }
        const int M = static_cast<int>(size.size());
        std::vector<int64_t> lab(M);
        std::iota(lab.begin(), lab.end(), int64_t{0});
        if (M < 2) {
            return lab;
        }
        std::vector<double> Dm(static_cast<size_t>(M) * M, 0.0);
        for (auto& [m, dw] : sp) { // sum(w * (1 - m @ m.T)) starting from 0
            const std::vector<double> G = gramRows(m->data(), M, dw.first);
            for (size_t q = 0; q < Dm.size(); q++) {
                Dm[q] = Dm[q] + dw.second * (1.0 - G[q]);
            }
        }
        for (int i = 0; i < M; i++) {
            Dm[static_cast<size_t>(i) * M + i] = INF;
        }
        std::vector<char> alive(M, 1);
        if (M > cap) { // only the largest groups merge freely
            std::vector<double> neg(M);
            for (int i = 0; i < M; i++) {
                neg[i] = -size[i];
            }
            const std::vector<int64_t> o = npArgsort(neg);
            for (int q = cap; q < M; q++) {
                const int64_t s = o[q];
                for (int t = 0; t < M; t++) {
                    Dm[static_cast<size_t>(s) * M + t] = INF;
                    Dm[static_cast<size_t>(t) * M + s] = INF;
                }
            }
        }
        // np.argmin(Dm) (the first minimum in row-major order) from per-row minima, updated as rows / columns change
        std::vector<double> rmin(M);
        std::vector<int> rarg(M);
        auto rowScan = [&](int r) {
            const double* row = &Dm[static_cast<size_t>(r) * M];
            int a = 0;
            for (int c = 1; c < M; c++) {
                if (row[c] < row[a]) {
                    a = c;
                }
            }
            rmin[r] = row[a];
            rarg[r] = a;
        };
        for (int r = 0; r < M; r++) {
            rowScan(r);
        }
        std::vector<double> d(M), g(M, 0.0);
        while (true) {
            int i = 0;
            for (int r = 1; r < M; r++) {
                if (rmin[r] < rmin[i]) {
                    i = r;
                }
            }
            const int j = rarg[i];
            const double dij = rmin[i];
            if (!std::isfinite(dij) || dij > tau2) {
                break;
            }
            const double si = size[i], sj = size[j];
            for (auto& [m, dw] : sp) {
                const int D = dw.first;
                for (int e = 0; e < D; e++) {
                    double& a = (*m)[static_cast<size_t>(i) * D + e];
                    a = (si * a + sj * (*m)[static_cast<size_t>(j) * D + e]) / (si + sj);
                }
            }
            size[i] = si + sj;
            alive[j] = 0;
            for (auto& l : lab) {
                if (l == j) {
                    l = i;
                }
            }
            for (int t = 0; t < M; t++) {
                Dm[static_cast<size_t>(j) * M + t] = INF;
                Dm[static_cast<size_t>(t) * M + j] = INF;
            }
            std::fill(d.begin(), d.end(), 0.0);
            for (auto& [m, dw] : sp) { // m @ m[i] (GEMV; M >= 2), only where the result is used
                const int D = dw.first;
                dgemvT(m->data(), M, D, m->data() + static_cast<size_t>(i) * D, g.data(), alive.data());
                for (int t = 0; t < M; t++) {
                    d[t] = d[t] + dw.second * (1.0 - g[t]);
                }
            }
            for (int t = 0; t < M; t++) {
                if (!alive[t]) {
                    d[t] = INF;
                }
            }
            d[i] = INF;
            for (int t = 0; t < M; t++) {
                Dm[static_cast<size_t>(i) * M + t] = d[t];
                Dm[static_cast<size_t>(t) * M + i] = d[t];
            }
            rowScan(i);
            rowScan(j);
            for (int t = 0; t < M; t++) {
                if (t == i || t == j) {
                    continue;
                }
                if (rarg[t] == i || rarg[t] == j) {
                    rowScan(t);
                } else if (d[t] < rmin[t] || (d[t] == rmin[t] && i < rarg[t])) {
                    rmin[t] = d[t];
                    rarg[t] = i;
                }
            }
        }
        return lab;
    }

    std::vector<int64_t> groupTracklet(const GroupPredictions& P, const std::vector<double>& on, double tau1, double tau2,
                                       double lam, std::vector<int64_t>* chain)
    {
        std::vector<int64_t> tr = groupChain(P, on, tau1, lam);
        if (chain != nullptr) {
            *chain = tr;
        }
        std::vector<int64_t> idx;
        for (int64_t i = 0; i < static_cast<int64_t>(tr.size()); i++) {
            if (tr[i] >= 0) {
                idx.push_back(i);
            }
        }
        if (idx.empty()) {
            return tr;
        }
        std::vector<int64_t> sub(idx.size()), tids, inv;
        for (size_t r = 0; r < idx.size(); r++) {
            sub[r] = tr[idx[r]];
        }
        unique(sub, tids, inv);
        const int M = static_cast<int>(tids.size());
        if (M == 1) {
            return tr;
        }
        const int E = P.E, K = P.K;
        const int64_t ni = static_cast<int64_t>(idx.size());
        std::vector<double> Pm(static_cast<size_t>(ni) * K);
        for (int64_t r = 0; r < ni; r++) {
            for (int k = 0; k < K; k++) {
                Pm[static_cast<size_t>(r) * K + k] = P.probs[static_cast<size_t>(idx[r]) * K + k];
            }
        }
        normaliseRows(Pm, ni, K);
        std::vector<double> size(M, 0.0);
        for (int64_t v : inv) {
            size[v] += 1.0;
        }
        std::vector<double> me(static_cast<size_t>(M) * E, 0.0), mp(static_cast<size_t>(M) * K, 0.0);
        for (int64_t r = 0; r < ni; r++) { // np.add.at, in order
            for (int e = 0; e < E; e++) {
                me[static_cast<size_t>(inv[r]) * E + e] += static_cast<double>(P.emb[static_cast<size_t>(idx[r]) * E + e]);
            }
            for (int k = 0; k < K; k++) {
                mp[static_cast<size_t>(inv[r]) * K + k] += Pm[static_cast<size_t>(r) * K + k];
            }
        }
        for (int t = 0; t < M; t++) {
            for (int e = 0; e < E; e++) {
                me[static_cast<size_t>(t) * E + e] /= size[t];
            }
            for (int k = 0; k < K; k++) {
                mp[static_cast<size_t>(t) * K + k] /= size[t];
            }
        }
        const std::vector<int64_t> lab = agglo({{std::move(me), 1.0}, {std::move(mp), lam}}, {E, K}, size, tau2);
        std::vector<int64_t> out(tr.size(), -1);
        for (int64_t r = 0; r < ni; r++) {
            out[idx[r]] = lab[inv[r]];
        }
        return out;
    }

    std::vector<double> trackStats(const std::vector<double>& on, const std::vector<double>& off, const std::vector<double>& pitch)
    {
        const int64_t n = static_cast<int64_t>(pitch.size());
        std::vector<double> p(n);
        for (int64_t i = 0; i < n; i++) {
            p[i] = pitch[i] / 64.0 - 1.0;
        }
        double poly = 0.0;
        if (n <= 3000) {
            std::vector<double> flag(n);
            for (int64_t i = 0; i < n; i++) {
                int64_t c = 0;
                for (int64_t j = 0; j < n; j++) {
                    c += std::min(off[i], off[j]) > std::max(on[i], on[j]) + 0.03;
                }
                flag[i] = c > 1.5 ? 1.0 : 0.0;
            }
            poly = npSum(flag.data(), n) / static_cast<double>(n);
        }
        const double mean = npSum(p.data(), n) / static_cast<double>(n);
        std::vector<double> sq(n);
        for (int64_t i = 0; i < n; i++) {
            const double x = p[i] - mean;
            sq[i] = x * x;
        }
        const double sd = std::sqrt(npSum(sq.data(), n) / static_cast<double>(n));
        return {mean, sd, *std::min_element(p.begin(), p.end()), *std::max_element(p.begin(), p.end()),
                std::log1p(static_cast<double>(n)) / 5, poly};
    }

    std::vector<double> trackLogits(const std::vector<int64_t>& ids, const GroupPredictions& P, const std::vector<double>& on,
                                    const std::vector<double>& off, const std::vector<double>& pitch,
                                    const std::vector<TrackHeadLayer>& th, double alpha)
    {
        const int K = P.K;
        std::vector<double> pn(K, 0.0);
        for (int64_t i : ids) { // .sum(0): row by row
            for (int k = 0; k < K; k++) {
                pn[k] = pn[k] + static_cast<double>(P.probs[static_cast<size_t>(i) * K + k]);
            }
        }
        const double ps = npSum(pn.data(), K);
        std::vector<double> ln(K);
        for (int k = 0; k < K; k++) {
            ln[k] = std::log(std::max(pn[k] / ps, 1e-9));
        }
        if (th.empty() || alpha <= 0 || P.h.empty()) {
            return ln;
        }
        const int D = P.D;
        std::vector<double> x(D, 0.0);
        for (int64_t i : ids) {
            for (int d = 0; d < D; d++) {
                x[d] = x[d] + static_cast<double>(P.h[static_cast<size_t>(i) * D + d]);
            }
        }
        for (int d = 0; d < D; d++) {
            x[d] = x[d] / static_cast<double>(ids.size());
        }
        std::vector<double> ton(ids.size()), toff(ids.size()), tp(ids.size());
        for (size_t q = 0; q < ids.size(); q++) {
            ton[q] = on[ids[q]];
            toff[q] = std::max(off[ids[q]], on[ids[q]] + 0.02);
            tp[q] = pitch[ids[q]];
        }
        const std::vector<double> st = trackStats(ton, toff, tp);
        x.insert(x.end(), st.begin(), st.end());
        const double sqrt2 = std::sqrt(2.0);
        for (size_t li = 0; li < th.size(); li++) {
            const TrackHeadLayer& L = th[li];
            if (static_cast<int>(x.size()) != L.in) {
                throw std::invalid_argument("track head: input size mismatch");
            }
            std::vector<double> y = matVec(L.w.data(), L.out, L.in, x.data());
            for (int o = 0; o < L.out; o++) {
                y[o] = y[o] + L.b[o];
            }
            if (li + 1 < th.size()) {
                for (double& v : y) {
                    v = (0.5 * v) * (1.0 + cephesErf(v / sqrt2));
                }
            }
            x.swap(y);
        }
        const double mx = *std::max_element(x.begin(), x.end());
        std::vector<double> lt(x.size()), ex(x.size());
        for (size_t k = 0; k < x.size(); k++) {
            lt[k] = x[k] - mx;
            ex[k] = std::exp(lt[k]);
        }
        const double lse = std::log(npSum(ex.data(), static_cast<int64_t>(ex.size())));
        std::vector<double> z(K);
        for (int k = 0; k < K; k++) {
            z[k] = alpha * (lt[k] - lse) + (1 - alpha) * ln[k];
        }
        return z;
    }

    std::vector<int64_t> absorbSmall(std::vector<int64_t> inv, const std::vector<double>& emb, int E, double frac, double tau_a)
    {
        const int64_t n = static_cast<int64_t>(inv.size());
        if (n == 0 || frac <= 0 || tau_a <= 0) {
            return inv;
        }
        std::vector<double> e = emb;
        normaliseRows(e, n, E);
        while (true) {
            std::vector<int64_t> tids, lab;
            unique(inv, tids, lab);
            const int M = static_cast<int>(tids.size());
            if (M < 2) {
                break;
            }
            std::vector<double> size(M, 0.0);
            for (int64_t v : lab) {
                size[v] += 1.0;
            }
            std::vector<double> cen(static_cast<size_t>(M) * E, 0.0);
            for (int64_t r = 0; r < n; r++) {
                for (int d = 0; d < E; d++) {
                    cen[static_cast<size_t>(lab[r]) * E + d] += e[static_cast<size_t>(r) * E + d];
                }
            }
            for (int t = 0; t < M; t++) {
                for (int d = 0; d < E; d++) {
                    cen[static_cast<size_t>(t) * E + d] /= size[t];
                }
            }
            bool done = true;
            for (int64_t t : npArgsort(size)) {
                const double s = size[t] / static_cast<double>(n);
                if (s >= frac) {
                    break;
                }
                std::vector<int64_t> big;
                for (int q = 0; q < M; q++) {
                    if (size[q] > size[t]) {
                        big.push_back(q);
                    }
                }
                if (big.empty()) {
                    continue;
                }
                const int nb = static_cast<int>(big.size());
                std::vector<double> cb(static_cast<size_t>(nb) * E);
                for (int b = 0; b < nb; b++) {
                    std::copy_n(&cen[static_cast<size_t>(big[b]) * E], E, &cb[static_cast<size_t>(b) * E]);
                }
                const std::vector<double> dots = matVec(cb.data(), nb, E, &cen[static_cast<size_t>(t) * E]);
                const double thr = tau_a * (1.0 - s / frac);
                double mn = INF;
                for (int b = 0; b < nb; b++) {
                    mn = std::min(mn, (1.0 - dots[b]) - thr);
                }
                if (mn < 0) {
                    std::vector<int64_t> mine;
                    for (int64_t r = 0; r < n; r++) {
                        if (lab[r] == t) {
                            mine.push_back(r);
                        }
                    }
                    normaliseRows(cb, nb, E);
                    std::vector<double> em(mine.size() * E);
                    for (size_t q = 0; q < mine.size(); q++) {
                        std::copy_n(&e[static_cast<size_t>(mine[q]) * E], E, &em[q * E]);
                    }
                    const std::vector<double> sc = matMulT(em.data(), static_cast<int>(mine.size()), E, cb.data(), nb);
                    for (size_t q = 0; q < mine.size(); q++) {
                        inv[mine[q]] = tids[big[argmaxFirst(&sc[q * nb], nb)]];
                    }
                    done = false;
                    break; // sizes / centres changed: start over
                }
            }
            if (done) {
                break;
            }
        }
        return inv;
    }
} // namespace

std::vector<std::pair<int, std::vector<int>>> groupDecode(const GroupPredictions& P, const std::vector<double>& on,
                                                          const std::vector<double>& off, const std::vector<double>& pitch,
                                                          const std::vector<TrackHeadLayer>& th, const GroupDecodeParams& prm,
                                                          const std::vector<double>& credit, GroupDecodeTrace* trace)
{
    const int n = static_cast<int>(on.size());
    if (n == 0) {
        return {};
    }
    if (P.n != n || static_cast<int>(off.size()) != n || static_cast<int>(pitch.size()) != n) {
        throw std::invalid_argument("groupDecode: note count mismatch");
    }
    const int E = P.E, K = P.K;
    if (static_cast<int64_t>(credit.size()) != static_cast<int64_t>(K) * K) {
        throw std::invalid_argument("groupDecode: CREDIT is not K x K");
    }
    std::vector<int64_t> chain;
    const std::vector<int64_t> track = groupTracklet(P, on, prm.tau, prm.tau2, prm.lam, &chain);
    if (trace != nullptr) {
        trace->tracklets = chain;
        trace->agglo = track;
    }
    std::vector<int64_t> idx;
    for (int i = 0; i < n; i++) {
        if (track[i] >= 0) {
            idx.push_back(i);
        }
    }
    if (trace != nullptr) {
        trace->idx = idx;
    }
    if (idx.empty()) {
        return {};
    }
    const int64_t ni = static_cast<int64_t>(idx.size());
    std::vector<int64_t> sub(ni), tids, inv;
    for (int64_t r = 0; r < ni; r++) {
        sub[r] = track[idx[r]];
    }
    unique(sub, tids, inv);
    const int T = static_cast<int>(tids.size());
    std::vector<int64_t> size = bincount(inv, T);
    const int64_t mn = prm.min_notes;
    std::vector<double> emb(static_cast<size_t>(ni) * E), pr(static_cast<size_t>(ni) * K);
    for (int64_t r = 0; r < ni; r++) {
        for (int d = 0; d < E; d++) {
            emb[static_cast<size_t>(r) * E + d] = P.emb[static_cast<size_t>(idx[r]) * E + d];
        }
        for (int k = 0; k < K; k++) {
            pr[static_cast<size_t>(r) * K + k] = P.probs[static_cast<size_t>(idx[r]) * K + k];
        }
    }
    for (int it = 0; it < prm.refine; it++) { // every note to the nearest big-track centre
        std::vector<int64_t> keepc;
        for (int t = 0; t < T; t++) {
            if (size[t] >= mn) {
                keepc.push_back(t);
            }
        }
        if (keepc.size() < 2) {
            break;
        }
        const int nk = static_cast<int>(keepc.size());
        std::vector<double> cen(static_cast<size_t>(T) * E, 0.0);
        for (int64_t r = 0; r < ni; r++) {
            for (int d = 0; d < E; d++) {
                cen[static_cast<size_t>(inv[r]) * E + d] += emb[static_cast<size_t>(r) * E + d];
            }
        }
        std::vector<double> ck(static_cast<size_t>(nk) * E);
        for (int q = 0; q < nk; q++) {
            std::copy_n(&cen[static_cast<size_t>(keepc[q]) * E], E, &ck[static_cast<size_t>(q) * E]);
        }
        normaliseRows(ck, nk, E);
        std::vector<double> sc = matMulT(emb.data(), static_cast<int>(ni), E, ck.data(), nk);
        if (prm.lam > 0) {
            std::vector<double> pm(static_cast<size_t>(T) * K, 0.0);
            for (int64_t r = 0; r < ni; r++) {
                for (int k = 0; k < K; k++) {
                    pm[static_cast<size_t>(inv[r]) * K + k] += pr[static_cast<size_t>(r) * K + k];
                }
            }
            std::vector<double> pk(static_cast<size_t>(nk) * K);
            for (int q = 0; q < nk; q++) {
                std::copy_n(&pm[static_cast<size_t>(keepc[q]) * K], K, &pk[static_cast<size_t>(q) * K]);
            }
            normaliseRows(pk, nk, K);
            std::vector<double> pnn = pr;
            normaliseRows(pnn, ni, K);
            for (double& v : pnn) { // `lam * (pn / ...) @ pm.T` parses as (lam * pn) @ pm.T
                v = prm.lam * v;
            }
            const std::vector<double> sp = matMulT(pnn.data(), static_cast<int>(ni), K, pk.data(), nk);
            for (size_t q = 0; q < sc.size(); q++) {
                sc[q] = sc[q] + sp[q];
            }
        }
        for (int64_t r = 0; r < ni; r++) {
            inv[r] = keepc[argmaxFirst(&sc[static_cast<size_t>(r) * nk], nk)];
        }
        size = bincount(inv, T);
    }
    if (trace != nullptr) {
        trace->refined = inv;
    }
    // fold small tracks into the best-matching big one
    std::vector<double> mass(static_cast<size_t>(T) * K, 0.0);
    for (int64_t r = 0; r < ni; r++) {
        for (int k = 0; k < K; k++) {
            mass[static_cast<size_t>(inv[r]) * K + k] += pr[static_cast<size_t>(r) * K + k];
        }
    }
    std::vector<int64_t> big;
    for (int t = 0; t < T; t++) {
        if (size[t] >= mn) {
            big.push_back(t);
        }
    }
    if (big.empty()) {
        big.push_back(argmaxFirst(size.data(), static_cast<int64_t>(size.size())));
    }
    std::vector<char> isbig(T, 0);
    for (int64_t b : big) {
        isbig[b] = 1;
    }
    const int nb = static_cast<int>(big.size());
    std::vector<double> bign(static_cast<size_t>(nb) * K);
    for (int q = 0; q < nb; q++) {
        std::copy_n(&mass[static_cast<size_t>(big[q]) * K], K, &bign[static_cast<size_t>(q) * K]);
    }
    normaliseRows(bign, nb, K);
    std::vector<double> src_v = mass;
    normaliseRows(src_v, T, K);
    for (int t = 0; t < T; t++) {
        if (!isbig[t] && size[t] > 0) {
            const std::vector<double> s = matVec(bign.data(), nb, K, &src_v[static_cast<size_t>(t) * K]);
            const int64_t to = big[argmaxFirst(s.data(), nb)];
            for (auto& v : inv) {
                if (v == t) {
                    v = to;
                }
            }
        }
    }
    if (trace != nullptr) {
        trace->folded = inv;
    }
    if (prm.absorb > 0 && prm.absorb_frac > 0) {
        inv = absorbSmall(inv, emb, E, prm.absorb_frac, prm.absorb);
    }
    if (trace != nullptr) {
        trace->absorbed = inv;
    }
    // naming: expected credit over the instruments, tracks of the same name merged (in first-insertion order)
    std::vector<int64_t> ts, dummy;
    unique(inv, ts, dummy);
    std::vector<std::pair<int, std::vector<int>>> out;
    std::map<int, size_t> pos;
    for (int64_t t : ts) {
        std::vector<int64_t> ids;
        for (int64_t r = 0; r < ni; r++) {
            if (inv[r] == t) {
                ids.push_back(idx[r]);
            }
        }
        const std::vector<double> z = trackLogits(ids, P, on, off, pitch, th, prm.alpha);
        int k;
        if (prm.ec) {
            const double zm = *std::max_element(z.begin(), z.end());
            std::vector<double> q(K);
            for (int c = 0; c < K; c++) {
                q[c] = std::exp(z[c] - zm);
            }
            const double qs = npSum(q.data(), K);
            for (int c = 0; c < K; c++) {
                q[c] = q[c] / qs;
            }
            const std::vector<double> ecr = matVec(credit.data(), K, K, q.data());
            k = static_cast<int>(argmaxFirst(ecr.data(), K));
        } else {
            k = static_cast<int>(argmaxFirst(z.data(), K));
        }
        auto it = prm.merge ? pos.find(k) : pos.end();
        if (it != pos.end()) {
            auto& dst = out[it->second].second;
            dst.insert(dst.end(), ids.begin(), ids.end());
        } else {
            if (prm.merge) {
                pos[k] = out.size();
            }
            out.emplace_back(k, std::vector<int>(ids.begin(), ids.end()));
        }
    }
    for (auto& [k, ids] : out) {
        std::sort(ids.begin(), ids.end());
    }
    return out;
}

} // namespace dtt
