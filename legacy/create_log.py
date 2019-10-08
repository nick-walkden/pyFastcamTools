def create_log(name,basic=False):
	import logging
	LOG_FILE = 'pyFastcamTools.log'
	logging.basicConfig(filename=LOG_FILE,level=logging.INFO,format='%(asctime)s : %(module)s : %(funcName)s : %(name)s : %(lineno)d ::: %(message)s' )
	logger = logging.getLogger(name)
	return logger