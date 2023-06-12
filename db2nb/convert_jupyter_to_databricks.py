# from https://github.com/Yoyodyne-Data-Science/ipynb-py-convert-databricks

import os
import re
import ipynb_convert_databricks as conv

def main():
   for root, dirs, files in os.walk("nbdev/", topdown=False):
      new_root = re.sub('^nbdev/', 'databricks/nbdev/', root)
      if os.path.exists(new_root) is False:
         os.makedirs(new_root)
      for name in files:
         ipynb_name = re.sub('.py', '.ipynb', name)
         conv.convert_databricks_nb(os.path.join(root, name), os.path.join(new_root, ipynb_name))

if __name__ == '__main__':
    main()