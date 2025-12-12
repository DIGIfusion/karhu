import pytest 
import os 
import sys 
import glob 

from karhu.cli.inference_from_helena import main as main_helena
TESTDIR = os.path.dirname(__file__)
TESTDATADIR = os.path.join(TESTDIR, "data")

CLIDIR  = os.path.join(TESTDIR, "..", "src", "karhu", "cli")
helena_folders = glob.glob(os.path.join(TESTDATADIR, "helena", "*"))
model_folders  = glob.glob(os.path.join(TESTDIR, "..", "model", "*"))
print(model_folders)
@pytest.mark.parametrize("helenadir", helena_folders)
# @pytest.mark.parametrize("modeldir", model_folders[0])
def test_run_with_cli(helenadir):
    modeldir = model_folders[0]
    # exe = f"{CLIDIR}/inference_from_helena.py" 
    # with pytest.MonkeyPatch.context() as m: 
    #     argv = ["python3", exe, f"-hd={helenadir}", f"-m={modeldir}"]
    #     m.setattr(sys, "argv", argv)
    main_helena(modeldir, helenadir)