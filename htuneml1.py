from threading import Thread
import requests, time,inspect, json
import traceback, datetime
class Job:
    def __init__(self, key=None, verbose=1):
        self.server = 'http://registru.ml/dlp'
        if key=='api_key':
            print('get the api key from '+self.server)
            assert False
        self.problems = {}
        #if 1 print the messages
        self.verbose = verbose
        self.current = None
        #list of parameters to set to the function
        self.params = None
        #stop the waiting for tasks and asking the server
        self.stop = False
        #key to the job id/problem id
        self.key = key
        self.running = False
        #the function to call inside the train function
        self.callb = None
        #name of the experiment, if none no logs are sent
        self.name = None
        self.keyExperiment = None
        self.stopEx = False
        self.expars = None
    def debug(self):
        # do not sent any logs to server if debug
        self.name = None
    def setName(self, name, pars=None):
        self.name=name        
        self.expars = pars
    def trymakenewD(self, values):        
        #get the pars of the function and try to make new experiments
        if self.running or self.name is None:
            return
        self.makeExperiment(self.name, values)
    def setFunction(self, n, p):
        # set the parameters for the function
        self.problems[n] = p
        if self.verbose == 1:
            print('new problem added')
    def setKey(self,k):
        self.key = k
        if self.verbose == 1:
            print('new key added')
    def waitTask(self, f):
        self.setFunction('f',f)
        self.stop = False
        if self.key is None:
            return 'no key'
        process = Thread(target=self.callServer, args=[])
        process.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop=True
            return 'ended'
        
    def sentEnd(self, status, mess):
        self.name = None
        if self.keyExperiment is None:
            print('no exp key')
            return
        res = requests.post(self.server, data={"action":'endJob','key':self.keyExperiment, 'status':status,
                                              'mess':mess})
        if res.ok:
            result = res.json()
    def callServer(self):
        while True:
            if self.stop:
                if self.verbose == 1:
                    print('stoped')
                break
            if self.params is None:
                time.sleep(5)
                res = requests.post(self.server, data={"action":'ask','key':self.key})
                if res.ok:
                    result = res.json()
                    print(result)
                    if self.verbose == 1:
                        print('waiting')
                    if result['new']:       
                        self.stopEx = False
                        if self.verbose == 1:
                            print('new problem starts')
                            print(result)
                        self.params = result['params']
                        self.current = result['current']
                        self.keyExperiment = result['key']
                    elif 'stop' in result and result['stop']:
                        self.stop = True
            else:
                self.running = True
                try:
                    self.problems[self.current](**self.params)
                    self.sentEnd('ended','')
                except Exception as e:
                    self.sentEnd('error',str(e))
                    print(e)
                    print(traceback.format_exc())
                self.running = False
                self.params = None

    def log(self, p):
        print(111,self.keyExperiment)
        if self.keyExperiment is None:
            if self.verbose == 1:
                print('no experiment key')
            return
        if 'type' not in p:
            p['type']=0
        res = requests.post(self.server, json={"action":'addI','key':self.keyExperiment, 'info':p})
        print(3)
        if res.ok:
            result = res.json()
            print(result)
            if result['stop']:
                self.stopEx = True
                
    def makeExperiment(self, name, params):
        self.stopEx = False        
        if self.key is None:
            if self.verbose == 1:
                print('no key')
            return
        data = {"action":'addEx','name':name,'nrExperiment':1,
                                               'w':0,'status':'running','gran':json.dumps(None),
                                               'mid':self.key, 'params':json.dumps(params)}
        if self.expars is not None:
            data.update(self.expars)
            
        res = requests.post(self.server, data=data)
        self.expars = None
        if self.verbose == 1:
            print('make experiment')
        if res.ok:
            result = res.json()
            self.keyExperiment = result['id']
            if self.verbose == 1:
                print('got key experimnet', result['id'])
    def getParams(self):
        return self.params#json.dump(self.params)
                            
    def setServer(self,server):
        self.server= server
    def getP(self, f):
        s = inspect.getargspec(f)
        p = {}
        for i in s.args:
            #print(i)
            p[i] = None
        n = len(s.args)-1 if len(s.defaults)!=len(s.args) else 0
        for i in s.defaults:
            #print(i)
            p[s.args[n]] = i
            n = n-1 if len(s.defaults)!=len(s.args) else n+1
        return p
    
    def sentParams(self, f):
        p = self.getP(f)
        #print(p)
        self.initparams = p
        #return
        res = requests.post(self.server,json={"action":'addP','key':self.key, 'params':p})
        if res.ok:
            print('sent',p.keys())
    def setCallBack(self, f):
        self.callb = f
    def monitor(self, func):
        def wrapper(*args, **kwargs):
            values = self.getP(func)
            s = inspect.getargspec(func)
            #print(args,s.args,kwargs, s.defaults)
            for i,k in enumerate(s.args): 
                if i>=len(args):continue
                values[k] = args[i]
            for k,v in  kwargs.items():
                values[k]=v
            #print(values)
            self.trymakenewD(values)
            if self.callb is not None and 'callback' in kwargs:
                kwargs['callback'] = kwargs
            if 'job' in kwargs:
                kwargs['job'] = self
            try:
                out = func(*args, **kwargs)
                self.sentEnd('ended','')
                return out
            except Exception as e:
                self.sentEnd('error',str(e))
                print(e)
                print(traceback.format_exc())
        return wrapper
