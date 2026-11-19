"""Fetch the four small benchmark sets (Bach10, Dagstuhl ChoirSet, PHENICX-Anechoic, RWC Royalty-Free + the RWC
annotations) with download.py: the four sets BENCHMARK.md uses (49 recordings, 89 min). Then build them with
    python py/bench/build_sets.py DOWNLOADS SETS bach10 dcs phenicx rwc_r

    python py/bench/download_subset.py DOWNLOADS
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download as D  # noqa: E402

root = sys.argv[1]
D.github_tree(D.BACH10, os.path.join(root, "bach10"))
D.github_tree(D.RWC_ANN, os.path.join(root, "rwc_annotations"))
url, md5 = D.zenodo_file(*D.PHENICX)
D.download(url, os.path.join(root, "phenicx", D.PHENICX[1]), md5)
url, md5 = D.zenodo_file(D.RWC[0], "RWC-R.zip")
D.download(url, os.path.join(root, "rwc", "RWC-R.zip"), md5)
url, _ = D.zenodo_file(D.DCS[0], D.DCS[1])
D.zip_members(url, os.path.join(root, "dcs"), D.DCS[2], "DagstuhlChoirSet_V1.2.3/")
print("SUBSET_DONE", flush=True)
