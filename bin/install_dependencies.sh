base_path=$PWD
relative_path="/env/bin/activate"
source $base_path$relative_path
pip install --upgrade setuptools pip
pip install -r requirements.txt