import os, sys
sys.path.insert(0, os.path.join(os.getcwd(), 'CODI'))
from tools.registry import registry
from dispatcher import Dispatcher
registry.load_all(mode='local')
print('tools', registry.list_names())
d = Dispatcher(registry)
res = d.dispatch({'action':'tool_call','tools':[{'name':'read_file','args':{'path':'README.md'}}]})
print(res)
