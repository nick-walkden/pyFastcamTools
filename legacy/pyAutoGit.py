"""
Function to automatically commit an analysis routine to an existing git repo
"""

try:
	from subprocess import call,Popen,PIPE
except:
	raise ImportError("ERROR: no subprocess module found. Ensure you are using python 2.4 or above")

class pyAutoGit(object):

	"""
	Class containing information and methods to commit to git automatically
	"""
	_commitFiles = None
	_LOGFILE = 'pyAutoGit.log'

	@staticmethod
	def init(files=[],logfile=None):
		import sys
		pyAutoGit.addFile(str(sys.argv[0]))
		if files:
			#Files to add to the commit
			for filename in files:
				pyAutoGit.addFile(filename)
				
		if not logfile==None:	
			pyAutoGit._LOGFILE = logfile
	
	@staticmethod
	def addFile(filename):
		
		if not pyAutoGit._commitFiles:
			pyAutoGit._commitFiles = filename+' '
		else:
			pyAutoGit._commitFiles += filename+' '
	
	@staticmethod					
	def removeFile(filename):
		import re
		#Remove the file from the commit string using regular expressions
		pyAutoGit._commitFiles = re.sub(filename+' ','',pyAutoGit._commitFiles)

	@staticmethod
	def setLogFile(filename):
		pyAutoGit._LOGFILE = filename

	@staticmethod
	def commit(log=True,logmessage="",commitmessage=""):
		import time
		
		addCom = 'git add '+pyAutoGit._commitFiles
		commitCom = 'git commit -m'
		commitMsg = ' \"pyAutoGit commit '+time.strftime("%x")+' '+time.strftime("%X")+' : '+commitmessage+'\"'

		#Call the add and commit commands
		call(addCom,shell=True)
		call(commitCom+commitMsg,shell=True)

		if log:
			import logging
			logging.basicConfig(filename=pyAutoGit._LOGFILE,level=logging.INFO,format='%(message)s' )
			log = logging.getLogger('pyAutoGitLogger')
			commitID = str(Popen('git rev-parse --short HEAD',shell=True,stdout=PIPE).communicate()[0])			
			logmsg = 'pyAutoGit commit '+commitID+time.strftime("%x")+' '+time.strftime("%X")+' :'+logmessage+'\nCommited Files: '+pyAutoGit._commitFiles[:]+'\n'
			log.info(logmsg)	





if __name__=='__main__':
	pyAutoGit.init()
	pyAutoGit.commit()


