#!/usr/bin/env python

import re
import numpy as np

"""
Geqdsk object to read data from using techniques from Ben Dudson

Nick Walkden, May 2015 

"""

def file_numbers(ingf):
	""" 
	Generator to read numbers in a file, originally written by Ben Dudson
	"""
	toklist = []
	while True:
		line = ingf.readline()
		if not line: break
		pattern = r'[+-]?\d*[\.]?\d+(?:[Ee][+-]?\d+)?'	#regular expression to find numbers
		toklist = re.findall(pattern,line)
		for tok in toklist:
			yield tok



class Geqdsk:   
	def __init__(self,filename=None):
		self.data = {}
		self.flags = {'loaded' : False }

		if filename != None:
			self.read(filename)
								
					
	def read(self,filename):
		""" 
		Read in data 
		"""
		
		if isinstance(filename, basestring):
			self._filename = filename	#filename is a string, so treat as filename
			self._file = open(filename)
		else:
			#assume filename is an object
			self._file = filename
			self._filename = str(filename)
		

		#First line should be case, id number and dimensions
		line = self._file.readline()
		if not line:
			raise IOError("ERROR: Cannot read from file"+self._filename)			
		
		conts = line.split() 	#split by white space
		self.data['nw'] = int(conts[-2])	
		self.data['nh'] = int(conts[-1])
		self.data['idum'] = int(conts[-3])

		self.flags['case'] = conts[0:-4]
	
		#Now use generator to read numbers
		token = file_numbers(self._file)
		
		float_keys = [
		'rdim','zdim','rcentr','rleft','zmid',
		'rmaxis','zmaxis','simag','sibry','bcentr',
		'current','simag','xdum','rmaxis','xdum',
		'zmaxis','xdum','sibry','xdum','xdum']
		
		#read in all floats
		for key in float_keys:		              			
			self.data[key] = float(token.next())
		
		#Now read arrays
		def read_1d(n):
			data = np.zeros([n])
			for i in np.arange(n):
				data[i] = float(token.next())
			return data

		def read_2d(nx,ny):
			data = np.zeros([nx,ny])
			for i in np.arange(nx):
				data[i,:] = read_1d(ny)
			return data

		

		
		self.data['fpol'] = read_1d(self.data['nw'])
		self.data['pres'] = read_1d(self.data['nw'])
		self.data['ffprime'] = read_1d(self.data['nw'])
		self.data['pprime'] = read_1d(self.data['nw'])
		self.data['psirz'] = read_2d(self.data['nw'],self.data['nh'])
		self.data['qpsi'] = read_1d(self.data['nw'])
	
		#Now deal with boundaries
		self.data['nbbbs'] = int(token.next())
		self.data['limitr'] = int(token.next())

		def read_bndy(nb,nl):
			if nb > 0:			
				rb = np.zeros(nb)
				zb = np.zeros(nb)
				for i in np.arange(nb):
					rb[i] = float(token.next())
					zb[i] = float(token.next())
			else:
				rb = [0]
				zb = [0]
		
			if nl > 0:
				rl = np.zeros(nl)
				zl = np.zeros(nl)
				for i in np.arange(nl):
					rl[i] = float(token.next())
					zl[i] = float(token.next())
			else:
				rl = [0]
				zl = [0]

			return rb,zb,rl,zl


		self.data['rbbbs'],self.data['zbbbs'],self.data['rlim'],self.data['zlim'] = read_bndy(self.data['nbbbs'],self.data['limitr'])
		
		self.flags['loaded'] = True
		

	def __getitem__(self,var):
		if self.flags['loaded']:
			return self.data[var]
		else:
			print "\nERROR: No gfile loaded"
			

	def get(self,var):
		if self.flags['loaded']:
			return self.data[var]
		else:
			print "\nERROR: No gfile loaded"


