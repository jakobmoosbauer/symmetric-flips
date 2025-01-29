# Fast matrix multiplication scheme search algorithm.
# Copyright (C) Mike Poole, September 2024.
# Contact: mikepoole73@googlemail.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# Version 22a0 - Implementation of the flip graph method, exploiting 3-way or 6-way symmetry.
# RELEASE VERSION MatrixMult22.py - based on mm22a0.

# This algorithm aims to find fast matrix multiplication schemes for square matrices in
# characteristic 2, using the flip graphs method.
# The flip and reduction operations used are as described in:
# M. Kauers and J. Moosbauer, 'Flip Graphs for Matrix Multiplication', 2022.
# The algorithm also uses the plus transition operation as described in:
# Y. Arai, Y. Ichikawa, K. Hukushima, 'Adaptive Flip Graph Algorithm for Matrix Multiplication', 2024.
# The algorithm aims to implement the basic flip operation as efficiently as possible, almost constant time, so 
# almost independent of the size of the problem (for practical runs) up to 8x8 size.
# The algorithm aims to reduce a symmetrized set of multiplications with either 3-way (cyclic) symmetry or 6-way
# (cyclic plus reflective) symmetry to the minimum number required by carrying out flip operations (in
# groups of 3 or 6 equivalent operations) which preserve the symmetry throughout.

# The script will use a fast, compiled C++ version of the solver if provided (full path given as the 
# variable 'fastsolver'), otherwise the slower Python routine is called.  

# New in version 20:
# - More efficient C++ solver - about 50% faster.
# - Option to suppress print output and write to log file.
# - Input file interface specified by adding a filename to the command line.
# - Allows continuation runs, taking saved solutions as starting points.

# New in version 21:
# - C++ solver improvements - more efficient method for implementing maximum length tabu.
# - Option to reset flip limit on every new lowest rank.
# - Option to set a rank limit on plus transitions.
# - Option to spread plus transitions out randomly instead of fixed separation.
# - Various bug fixes and tidies.

# New in version 22:
# - Bespoke dictionary data structure class for C++ solver - significantly more efficient.
# - More efficient generation of random numbers and selection of flip in C++ solver.
# - Option to split early termination to allocate a percentage of resources to achieving a certain rank.

import random
import time
import copy
import glob
import os
import subprocess
import datetime
import sys

matdim=4
runcase=1
matsize=matdim*matdim
matvecs=2**matsize
row=[[0]*matsize for i in range(3)]
col=[[0]*matsize for i in range(3)]
odr=[[0]*matsize for i in range(3)]
fastsolver='C:/Flip Graphs/FlipSolver22/x64/Release/FlipSolver22.exe'
if not os.path.isfile(fastsolver): fastsolver=None

ctrls=[		# Globally available editable controls.
0,			# 0 - used to store run number.
0,			# 1 - termination strategy, 0 at flip limit, 1 early termination, 2 rolling flip limit.
3000000,	# 2 - maximum number of flips before abort.
0,			# 3 - used to store random seed.
0,			# 4 - split percent.
1,			# 5 - number of solves.
0,			# 6 - print multiplication set, 0 full, 1 one-line summary.
1,			# 7 - flag for print options, -1, none, 0 summary, 1 standard, 2 detailed, 3 diagnostic.
1,			# 8 - write run outcomes to log file.
1,			# 9 - turn off matplotlib plots.
0,			# 10 - used for plotting variable.
0,			# 11 - used for plotting variable.
6000,		# 12 - plus transition frequency, 0 never.
0,			# 13 - plus transition spacing, 0 uniform, 1 random sample from 2*plus frequency.
0,			# 14 - maximum size, 0 umlimited, +ve limit on A*B*C, -ve limit on A, B and C.
0,			# 15 - plus transition limit, 0 size of problem.
0,			# 16 - unused.
0,			# 17 - frequency for plot evolution stored in ctrls[10] (Python solver only).
0,			# 18 - unused.
0,			# 19 - unused.
0]			# 20 - unused.

if ctrls[9]==0:
	import matplotlib.pyplot as plt
	import numpy as np

def main():
	'''Fast matrix multiplication search algorithm - main program.'''
	if len(sys.argv)>1: inputfile(sys.argv[1]); return

	# Premilinaries.
	if ctrls[7]>=0: print('Fast matrix multiplication search algorithm by Mike Poole - version 22.')
	tt=time.time()
	rseed=int(1000000*tt+1000000*os.getpid())%10000000000
#	rseed=0
	ctrls[3]=rseed
	random.seed(rseed)
	if ctrls[7]>=0: print('Random number seed:',ctrls[3])
	setrco(1)
	answer()
	if ctrls[7]>=0: print('Solution:',matstr(answ))

	# Run cases.
	runmanager()

	# Wrap up.
	if ctrls[7]>=0: print('\nTotal runs:',ctrls[0])
	tt=time.time()-tt
	if ctrls[7]>=0: print(); print('Run complete - CPU time:',f'{tt:.2f}','seconds'); print()

def inputfile(iname):
	'''Read input file and run cases as detailed there.'''
	global matdim,matsize,matvecs,row,col,odr

	# Read input file and override global settings.
	if not os.path.exists(iname): print('Input file',iname,'not found.'); return
	flags=0
	fname=None
	start=0
	diagc=None
	fullc=None
	with open(iname,'r') as f:
		lines=f.readlines()
		for l in lines:
			a=l.split()
			if len(a)>0:
				if a[0]!='#':
					if a[0]=='MATRIX_SIZE:': matdim=int(a[1]); flags|=1<<0
					if a[0]=='FLIP_LIMIT:': ctrls[2]=int(a[1]); flags|=1<<1
					if a[0]=='TERMINATION_STRATEGY:':
						if a[1]=='LIMIT': ctrls[1]=0; flags|=1<<2
						elif a[1]=='EARLY': ctrls[1]=1; flags|=1<<2
						elif a[1]=='RESET': ctrls[1]=2; flags|=1<<2
						elif a[1]=='SPLIT': ctrls[1]=int(a[2]); ctrls[4]=int(a[3]); flags|=1<<2
					if a[0]=='PLUS_TRANSITION_AFTER:': ctrls[12]=int(a[1]); flags|=1<<3
					if a[0]=='PLUS_TRANSITION_LIMIT:': ctrls[15]=int(a[1]); flags|=1<<4
					if a[0]=='NUMBER_OF_SOLVES:': ctrls[5]=int(a[1]); flags|=1<<5
					if a[0]=='PRINT_OUTPUT:':
						if a[1]=='NONE': ctrls[7]=-1; flags|=1<<6
						elif a[1]=='SUMMARY': ctrls[7]=0; flags|=1<<6
						elif a[1]=='STANDARD': ctrls[7]=1; flags|=1<<6			
						elif a[1]=='DETAILED': ctrls[7]=2; flags|=1<<6
						elif a[1]=='DIAGNOSTIC': ctrls[7]=3; flags|=1<<6
					if a[0]=='SCHEME_STYLE:':
						if a[1]=='FULL': ctrls[6]=0; flags|=1<<7
						elif a[1]=='SUMMARY': ctrls[6]=1; flags|=1<<7
					if a[0]=='WRITE_LOG:':		
						if a[1]=='NO': ctrls[8]=0; flags|=1<<8
						elif a[1]=='YES': ctrls[8]=1; flags|=1<<8
					if a[0]=='SAVE:':
						if a[1]=='ALL': save=-1; flags|=1<<9
						else: save=int(a[1]); flags|=1<<9
					if a[0]=='RANDOM_SEED:':
						if a[1]=='AUTO': rseed=-1; flags|=1<<10
						else: rseed=int(a[1]); flags|=1<<10
					if a[0]=='PLUS_TRANSITION_RANDOM:':
						if a[1]=='NO': ctrls[13]=0; flags|=1<<11
						elif a[1]=='YES': ctrls[13]=1; flags|=1<<11
					if a[0]=='MAXIMUM_SIZE:':
						if a[1]=='NONE': ctrls[14]=0; flags|=1<<12
						elif a[1]=='LENGTH': ctrls[14]=-int(a[2]); flags|=1<<12
						elif a[1]=='VOLUME': ctrls[14]=int(a[2]); flags|=1<<12
					if a[0]=='RUN_TYPE:':
						if a[1]=='NEW': rt=0; flags|=1<<13
						elif a[1]=='CONTINUATION': rt=1; flags|=1<<13
					if a[0]=='TARGET:': target=int(a[1]); flags|=1<<14
					if a[0]=='SYMMETRY:': symm=int(a[1]); flags|=1<<15
					if a[0]=='SAVED_FILE:': fname=a[1]
					if a[0]=='SAVED_SIZE:':
						if a[1]=='RANDOM': start=-1
						else: start=int(a[1])
					if a[0]=='DIAGONAL_CUBES:':
						i=1;
						diagc=[]
						while len(a[i])==matdim:
							diagc.append(a[i])
							if i==len(a)-1: break
							i+=1
					if a[0]=='FULL_CUBES:':			
						i=1;
						fullc=[]
						while len(a[i])==matdim*matdim:
							fullc.append(a[i])
							if i==len(a)-1: break
							i+=1
					if a[0]=='PLUS_TRANSITION_HEADROOM:': print('Keyword PLUS_TRANSITION_HEADROOM: withdrawn.'); return
					if a[0]=='PLUS_TRANSITION_CAP:': print('Keyword PLUS_TRANSITION_CAP: withdrawn.'); return
					if a[0]=='EARLY_TERMINATION:': print('Keyword EARLY_TERMINATION: withdrawn.'); return
					
	if flags!=65535: print('Missing input:',bin(flags)[2:]); return
	if rt==1 and start==0 and fname==[]: print('Error in input file.'); return

	# Set global size data.
	matsize=matdim*matdim
	matvecs=2**matsize
	row=[[0]*matsize for i in range(3)]
	col=[[0]*matsize for i in range(3)]
	odr=[[0]*matsize for i in range(3)]

	# Premilinaries.
	if ctrls[7]>=0: print('Fast matrix multiplication search algorithm by Mike Poole - version 22.')
	tt=time.time()
	if rseed==-1: rseed=int(1000000*tt+1000000*os.getpid())%10000000000
	ctrls[3]=rseed
	random.seed(rseed)
	if ctrls[7]>=0: print('Random number seed:',ctrls[3])
	setrco(1)
	answer()
	if ctrls[7]>=0: print('Solution:',matstr(answ))

	# Run cases.
	ctrls[11]=[0]*1000
	for r in range(ctrls[5]):
		ctrls[0]+=1
		if rt==0:
			if fullc!=None: mset=standardrun(fullc=fullc,target=target,symm=symm,save=save)
			elif diagc!=None: mset=standardrun(diagc=diagc,target=target,symm=symm,save=save)
			else: mset=standardrun(target=target,symm=symm,save=save)
		elif rt==1:
			if fname==None: mset=runfromfile(start=start,target=target,symm=symm,save=save)
			else: mset=runfromfile(fname=fname,target=target,symm=symm,save=save)

	# Summary output.
	if ctrls[7]>=0:
		s='Summary:'
		for i in range(1000):
			if ctrls[11][i]>0: s+=' '+str(i)+'/'+str(ctrls[11][i])
		print(s)
	if ctrls[8]==1:
		with open('runlog.txt','a') as f:
			s=str(ctrls[3]).zfill(10)+' Summary:'
			for i in range(1000):
				if ctrls[11][i]>0: s+=' '+str(i)+'/'+str(ctrls[11][i])
			s+='\n'
			f.write(s)
		
	# Wrap up.
	if ctrls[7]>=0: print('\nTotal runs:',ctrls[0])
	tt=time.time()-tt
	if ctrls[7]>=0: print(); print('Run complete - CPU time:',f'{tt:.2f}','seconds'); print()

def runmanager():
	'''Step through the logic and report output for different run types.'''

	# Series of standard runs, or runs from file.
	ctrls[11]=[0]*1000
	for r in range(ctrls[5]):
		ctrls[0]+=1
		if matdim==2: # Suggest ctrls[2]=10000, ctrls[12]=0.
			if runcase==0: # Strassen algorithm, 3-way symmetry.
				mset=standardrun(diagc=['11'],target=7,symm=3)
			if runcase==1: # Strassen algorithm, 6-way symmetry.
				mset=standardrun(diagc=['11'],target=7,symm=6)
			if runcase==2: # Winograd variant of Strassen algorithm, 3-way symmetry.
				mset=standardrun(fullc=['1000','0111','0101','0011'],target=7,symm=3)
			if runcase==3: # 0-cube, 3-way symmetry.
				mset=standardrun(diagc=[],target=9,symm=3)
		if matdim==3: # Suggest ctrls[2]=300000, ctrls[12]=1000.
			if runcase==0: # 1-cube 25 multiplications, 3-way symmetry.
				mset=standardrun(diagc=['111'],target=25,symm=3)
			if runcase==1: # 1-cube 25 multiplications, 6-way symmetry.
				mset=standardrun(diagc=['111'],target=25,symm=6)
			if runcase==2: # 2-cube algorithm, 3-way symmetry.
				mset=standardrun(fullc=['100001001','000011000'],target=23,symm=3)
			if runcase==3: # 5-cube algorithm (similar to Laderman scheme), 3-way symmetry.
				mset=standardrun(fullc=['100000000','010110000','010010000','000110000','000000001'],target=23,symm=3)
			if runcase==4: # 11-cube algorithm, 3-way symmetry.
				mset=standardrun(fullc=['100000000','010110000','010010000','000110000','001101011','001001001',
				'001001011','000101101','000101011','001101101','000000011'],target=23,symm=3)
			if runcase==5: # 2-cube algorithm (simplest), 3-way symmetry.
				mset=standardrun(diagc=['010','101'],target=23,symm=3)
			if runcase==6: # 0-cube, 3-way symmetry.
				mset=standardrun(diagc=[],target=27,symm=3)
		if matdim==4: # Suggest ctrls[2]=3000000, ctrls[12]=3000 if symm=3, 6000 if symm=6.
			if runcase==0: # 1-cube, 3-way symmetry.
				mset=standardrun(diagc=['1111'],target=49,symm=3)
			if runcase==1: # 1-cube, 6-way symmetry.
				mset=standardrun(diagc=['1111'],target=49,symm=6,save=49)
			if runcase==2: # 2-cube, 6-way symmetry.
				mset=standardrun(diagc=['1001','0110'],target=50,symm=6)
			if runcase==3: # 4-cube, 6-way symmetry.
				mset=standardrun(diagc=['1000','0100','0010','0001'],target=52,symm=6)
			if runcase==4: # 0-cube, 3-way symmetry.
				mset=standardrun(diagc=[],target=51,symm=3)
		if matdim==5: # Suggest ctrls[2]=300000000, ctrls[12]=50000 if symm=3, 100000 if symm=6.
			if runcase==0: # 1-cube, 3-way symmetry.
				mset=standardrun(diagc=['11111'],target=97,symm=3,save=97)
			if runcase==1: # 1-cube, 6-way symmetry.
				mset=standardrun(diagc=['11111'],target=97,symm=6,save=97)
			if runcase==2: # 2-cube, 3 way symmetry.
				mset=standardrun(diagc=['11011','00100'],target=95,symm=3,save=95)
			if runcase==3: # 3-cube, 3 way symmetry.
				mset=standardrun(diagc=['10001','01010','00100'],target=93,symm=3,save=93)
			if runcase==4: # 4-cube, 3 way symmetry.
				mset=standardrun(diagc=['10001','01000','00100','00010'],target=94,symm=3,save=94)
			if runcase==5: # 3-cube, 3 way symmetry, run to hold point.
				mset=standardrun(diagc=['10001','01010','00100'],target=105,symm=3,save=105)
			if runcase==6: # 3-cube, 3 way symmetry, run from hold point.
				mset=runfromfile(start=105,target=93,symm=3,save=93)
		if matdim==6: # Suggest ctrls[2]=10000000000, ctrls[12]=500000 if symm=3, 1000000 if symm=6.
			if runcase==0: # 1-cube, 3-way symmetry.
				mset=standardrun(diagc=['111111'],target=163,symm=3,save=163)
			if runcase==1: # 1-cube, 6-way symmetry.
				mset=standardrun(diagc=['111111'],target=163,symm=6,save=163)
			if runcase==2: # 2-cube, 3-way symmetry.
				mset=standardrun(diagc=['110011','001100'],target=161,symm=3,save=161)
			if runcase==3: # 2-cube, 3-way symmetry.
				mset=standardrun(diagc=['110011','001100'],target=188,symm=3,save=188)
			if runcase==4: # 0-cube, 6-way symmetry.
				mset=standardrun(diagc=[],target=168,symm=6,save=168)
		if matdim==7:
			if runcase==1: # 1-cube, 6-way symmetry.
				mset=standardrun(diagc=['1111111'],target=163,symm=6,save=163)

	# Overall summary.
	if ctrls[7]>=0:
		s='Summary:'
		for i in range(1000):
			if ctrls[11][i]>0: s+=' '+str(i)+'/'+str(ctrls[11][i])
		print(s)
	if ctrls[8]==1:
		with open('runlog.txt','a') as f:
			s=str(ctrls[3]).zfill(10)+' Summary:'
			for i in range(1000):
				if ctrls[11][i]>0: s+=' '+str(i)+'/'+str(ctrls[11][i])
			s+='\n'
			f.write(s)	

def standardrun(diagc=None,fullc=None,target=0,symm=3,save=0):
	'''Carry out one standard run.'''

	# First set up cubes:
	cubes=[]
	if fullc!=None:
		for fc in fullc:
			fcl=[y for y in range(matsize) if fc[y]=='1']
			x=convert(fcl)
			cube=[x,x,x]
			cubes.append(cube)
	else:
		for dc in diagc:
			dcl=[y*matdim+y for y in range(matdim) if dc[y]=='1']
			x=convert(dcl)
			cube=[x,x,x]
			cubes.append(cube)
	l=len(cubes)

	# If first solve, report details.
	if ctrls[0]==1:
		s='Size: '+str(matdim)+' Cubes: '+str(l)+' Target: '+str(target)+' Symm: '+str(symm)
		if save>0: s+=' Save <=: '+str(save)
		if save==-1: s+=' Save: All'
		t='Flip limit: '+str(ctrls[2])
		if ctrls[1]==1: t+='(E)'
		if ctrls[1]==2: t+='(R)'
		if ctrls[1]>2: t+='(S'+str(ctrls[1])+':'+str(ctrls[4])+'%)'
		if ctrls[12]>0: t+=' Plus after: '+str(ctrls[12])
		if ctrls[13]==1 and ctrls[12]>0: t+='(R)'
		if ctrls[15]>0: t+=' Plus limit: '+str(ctrls[15])
		if ctrls[18]>0: t+=' Super: '+str(ctrls[18])+'/'+str(ctrls[19])
		if ctrls[14]<0: t+=' Maximum length: '+str(-ctrls[14])
		elif ctrls[14]>0: t+=' Maximum volume: '+str(ctrls[14])
		if ctrls[7]>=0:
			print('New run -',s)
			print(t)
		if ctrls[8]==1:
			with open('runlog.txt','a') as f:
				now=datetime.datetime.now()
				nowstr=now.strftime('%d/%m/%Y %H:%M:%S')
				s=' Run at: '+nowstr+' '+s+'\n'
				t=' '+t+'\n'
				f.write(str(ctrls[3]).zfill(10)+s)
				f.write(str(ctrls[3]).zfill(10)+t)

	# Set up multiplication set and solve.
	dset=MultSet()
	dset.muls=cubes
	dset.nomuls=l
	dset.evalall()
	psymm=6; pl=matsize**3
	z=bin(dset.curr)[2:].zfill(pl)
	for i in range(pl):
		if z[i]!=z[pl-1-i]: psymm=3; break
	mset=MultSet(pattern=dset.curr,symm=psymm)
	start=mset.nomuls
	if ctrls[15]==0: headroom=0
	else: headroom=ctrls[15]-l-mset.nomuls
	headroom-=headroom%symm
	if headroom>0:
		for i in range(headroom): mset.muls.append([0,0,0]); mset.nomuls+=1; mset.maxplus+=1
	elif headroom<0: mset.maxplus+=headroom
	code,mmin,st=mset.solve(target-l,l,symm)
	best=mmin+l
	mset=MultSet(orig=mset)
	for m in dset.muls: mset.muls.append(m); mset.nomuls+=1
	mset.evalall()

	# Save results, and print.
	if best<=save or (save==-1 and best<start):
		if not os.path.exists('results'): os.mkdir('results')
		rf=random.randrange(10000000000)
		while True:
			fname='results/m'+str(best).zfill(3)+'r'+str(rf).zfill(10)+'.txt'
			if not os.path.exists(fname): break
			rf+=1
		mset.writesol(fname)	
	if ctrls[17] and fastsolver==None:
		ctrls[10]=[x+l for x in ctrls[10]]
		plotres(ctrls[10])
	ctrls[11][best]+=1
	if ctrls[7]>=0: print('Run:',ctrls[0],'Best:',best,st)
	if ctrls[8]==1:
		with open('runlog.txt','a') as f:
			s=str(ctrls[3]).zfill(10)+'/'+str(ctrls[0]).zfill(3)+' Best: '+str(best)+' '+st+'\n'
			f.write(s)
	if best==target:
		if ctrls[7]>=1: print(mset)
		return mset
	else:
		if ctrls[7]>=2: print(mset)
		return None

def runfromfile(fname=None,start=0,target=0,symm=3,save=0):
	'''Load initial guess from file and then run.'''
	hname='results/history.txt'

	# Choose file as start point.
	if fname==None:
		if start==0: print('Need either filename or specified start point.'); return
		if start==-1: 
			rfiles='results/m*.txt'
			fnames=glob.glob(rfiles)
			if len(fnames)==0: print('No saved solutions exist.'); return
			fname=random.choice(fnames)
			start=int(fname[9:12])
		else:
			rfiles='results/m'+str(start).zfill(3)+'*.txt'
			fnames=glob.glob(rfiles)
			if len(fnames)==0: print('No saved solutions at',start,'exist.'); return
			fname=random.choice(fnames)
	else:
		fname='results/'+fname
		start=int(fname[9:12])

	# Load chosen result.
	fset=MultSet(fname=fname)
	dset=MultSet()
	mset=MultSet()
	for m in fset.muls:
		if m[0]==m[1]==m[2]: dset.muls.append(m); dset.nomuls+=1
		else: mset.muls.append(m); mset.nomuls+=1
	l=dset.nomuls
	mset.maxplus=mset.nomuls

	# If first solve, report details.
	if ctrls[0]==1:
		s='Size: '+str(matdim)+' Cubes: '+str(l)+' Target: '+str(target)+' Symm: '+str(symm)
		if save>0: s+=' Save <=: '+str(save)
		if save==-1: s+=' Save: All'
		t='Flip limit: '+str(ctrls[2])
		if ctrls[1]==1: t+='(E)'
		if ctrls[1]==2: t+='(R)'
		if ctrls[1]>2: t+='(S'+str(ctrls[1])+')'
		if ctrls[12]>0: t+=' Plus after: '+str(ctrls[12])
		if ctrls[13]==1 and ctrls[12]>0: t+='(R)'
		if ctrls[15]>0: t+=' Plus limit: '+str(ctrls[15])
		if ctrls[18]>0: t+=' Super: '+str(ctrls[18])+'/'+str(ctrls[19])
		if ctrls[14]<0: t+=' Maximum length: '+str(-ctrls[14])
		elif ctrls[14]>0: t+=' Maximum volume: '+str(ctrls[14])
		if ctrls[7]>=0:
			print('Continuation run -',s)
			print(t)
		if ctrls[8]==1:
			with open('runlog.txt','a') as f:
				now=datetime.datetime.now()
				nowstr=now.strftime('%d/%m/%Y %H:%M:%S')
				s=' Run at: '+nowstr+' '+s+'\n'
				t=' '+t+'\n'
				f.write(str(ctrls[3]).zfill(10)+s)
				f.write(str(ctrls[3]).zfill(10)+t)

	# Set up multiplications and solve.
	if ctrls[15]==0: headroom=0
	else: headroom=ctrls[15]-l-mset.nomuls
	headroom-=headroom%symm
	if headroom>0:
		for i in range(headroom): mset.muls.append([0,0,0]); mset.nomuls+=1; mset.maxplus+=1
	elif headroom<0: mset.maxplus+=headroom
	code,mmin,st=mset.solve(target-l,l,symm)	
	best=mmin+l
	mset=MultSet(orig=mset)
	for m in dset.muls: mset.muls.append(m); mset.nomuls+=1
	mset.evalall()

	# Save results if necessary, overwrite start file if no improvement, and print.
	if best<=save or (save==-1 and best<=start):
		if not os.path.exists('results'): os.mkdir('results')
		if best==start: sname=fname
		else:
			rf=random.randrange(10000000000)
			while True:
				sname='results/m'+str(best).zfill(3)+'r'+str(rf).zfill(10)+'.txt'
				if not os.path.exists(sname): break
				rf+=1
		mset.writesol(sname)
	if ctrls[17] and fastsolver==None:
		ctrls[10]=[x+l for x in ctrls[10]]
		plotres(ctrls[10])
	ctrls[11][best]+=1
	if ctrls[7]>=0: print('Run:',ctrls[0],'From:',fname[8:],'Best:',best,st)
	if ctrls[8]==1:
		with open('runlog.txt','a') as f:
			s=str(ctrls[3]).zfill(10)+'/'+str(ctrls[0]).zfill(3)+' From: '+fname[8:]+' Best: '+str(best)+' '+st+'\n'
			f.write(s)

	# Update history file.
	with open(hname,'a') as f:			
		s=fname+' '+str(start)+' '+str(best)+' '+str(mset.flips)+'\n'
		f.write(s)

	# Print if necessary and return.
	if best==target:
		if ctrls[7]>=1: print(mset)
		return mset
	else:
		if ctrls[7]>=2: print(mset)
		return None

class MultSet:
	'''Object representing a set of multiplications.'''
	# Version 7.10:
	# Bespoke version for mm19 - flip graphs method.

	inst=0

	def __init__(self,fname=None,orig=None,save=False,pattern=None,symm=1):
		'''MultSet constructor.'''
		MultSet.inst+=1
		self.id=MultSet.inst
		self.base=answ
		self.curr=0
		self.err=0
		self.flips=0
		self.nomuls=0
		self.maxplus=0
		self.muls=[]
		self.ring=None

		# Load scheme from file.
		if fname!=None:
			self.loadsol(fname)
			self.evalall()

		# Copy scheme from a MultSet object.
		if orig!=None:
			self.nomuls=0
			self.muls=[]
			for m in orig.muls:
				if m[0]*m[1]*m[2]!=0:
					self.muls.append(m)
					self.nomuls+=1
			self.flips=orig.flips
			self.evalall()

		# Define scheme from pattern representing current error, with symmetry option.
		elif pattern!=None:
			if symm==1:
				self.muls=[]
				for a in range(matsize):
					for b in range(matsize):
						for c in range(matsize):
							d=a+matsize*b+matsize*matsize*c
							if pattern&1<<d:
								self.muls.append([1<<a,1<<b,1<<c])
								self.nomuls+=1
			elif symm==3:
				left=pattern
				self.muls=[]
				for a in range(matsize):
					for b in range(matsize):
						for c in range(matsize):
							d=a+matsize*b+matsize*matsize*c
							if left&1<<d:
								self.muls.append([1<<a,1<<b,1<<c])
								left^=1<<d
								d=c+matsize*a+matsize*matsize*b
								self.muls.append([1<<c,1<<a,1<<b])
								left^=1<<d
								d=b+matsize*c+matsize*matsize*a
								self.muls.append([1<<b,1<<c,1<<a])
								left^=1<<d
								self.nomuls+=3
			elif symm==6:
				self.muls=[]
				left=pattern
				for a in range(matsize):
					for b in range(matsize):
						for c in range(matsize):
							d=a+matsize*b+matsize*matsize*c
							if left&1<<d:
								self.muls.append([1<<a,1<<b,1<<c])
								left^=1<<d
								d=c+matsize*a+matsize*matsize*b
								self.muls.append([1<<c,1<<a,1<<b])
								left^=1<<d
								d=b+matsize*c+matsize*matsize*a
								self.muls.append([1<<b,1<<c,1<<a])
								left^=1<<d
								ma=matsize-a-1; mb=matsize-b-1; mc=matsize-c-1
								d=ma+matsize*mb+matsize*matsize*mc
								self.muls.append([1<<ma,1<<mb,1<<mc])
								left^=1<<d
								d=mc+matsize*ma+matsize*matsize*mb
								self.muls.append([1<<mc,1<<ma,1<<mb])
								left^=1<<d
								d=mb+matsize*mc+matsize*matsize*ma
								self.muls.append([1<<mb,1<<mc,1<<ma])
								left^=1<<d
								self.nomuls+=6

			self.maxplus=self.nomuls
			self.evalall()
		if save: sname=str(matdim)+'-m'+str(self.nomuls)+'.txt'; self.writesol(sname)

	def __del__(self):
		'''MultSet destructor.'''
		MultSet.inst-=1

	def __str__(self):
		'''MultSet print.'''
		if ctrls[6]==1:
			s=''
			sz=[]
			for m in self.muls:
				lv=len(m)
				if lv==1: si=m[0].bit_count()**3
				elif lv==2: si=m[0].bit_count()*m[1].bit_count()**2
				else: si=m[0].bit_count()*m[1].bit_count()*m[2].bit_count()
				sz.append(si)
			ssz='[ '
			for z in sz: ssz+=str(z)+' '
			ssz+=']'
			s+=str(ctrls[0])+' Muls: '+str(self.nomuls)+' '+ssz+' Flips: '+str(self.flips)
			s+=' Error: '+str(self.err)
			return s
		s='\n'
		if matdim<=6:
			if self.nomuls>0:
				s+='Multiplication set ('+str(self.nomuls)+'):\n'
				s+='R: |'
				for i in range(matsize): s+=str(row[0][i]+1)
				s+='|    |'
				for i in range(matsize): s+=str(row[1][i]+1)
				s+='|    |'
				for i in range(matsize): s+=str(row[2][i]+1)
				s+='|\nC: |'
				for i in range(matsize): s+=str(col[0][i]+1)
				s+='|    |'
				for i in range(matsize): s+=str(col[1][i]+1)
				s+='|    |'
				for i in range(matsize): s+=str(col[2][i]+1)
				s+='|\n'
				d=matsize*3+31
				for i in range(d): s+='-'
				s+='\n'
			for i in range(self.nomuls):
				t=self.entrstr(i)
				s+=t+'\n'
		s+=matstr(self.curr)
		s+='Run: '+str(ctrls[0])+' Flips: '+str(self.flips)+' Error: '+str(self.err)
		s+='\n'
		return s

	def solve(self,target,cubes,symm=3):
		'''Interface to solver to reduce number of multiplications.'''
		tt=time.time()
		self.flips=0
		rcode=9
		flimit=ctrls[2]
		plimit=ctrls[12]
		if ctrls[13]==1: plimit=-plimit
		maxsize=ctrls[14]
		if plimit==0: plimit=flimit*1007
		termination=ctrls[1]
		if termination>2: termination-=cubes; termination-=termination%symm
		split=ctrls[4]
		if target<0: target=self.nomuls+target
		rseed=random.randrange(1000000000)
		iname='int'+str(ctrls[3]).zfill(10)+'.txt'
		with open(iname,'w') as f:
			s=str(self.nomuls)+' '+str(self.flips)+' '+str(rcode)+' '+str(target)+' '+str(flimit)+' '
			s+=str(plimit)+' '+str(termination)+' '+str(rseed)+' '+str(symm)+' '+str(self.maxplus)+' '
			s+=str(split)+' '+str(self.nomuls)+' '+str(maxsize)+'\n'
			f.write(s)
			for m in self.muls: s=str(m[0])+'\n'; f.write(s)
		if fastsolver==None: flipsolver(iname)
		else: subprocess.run([fastsolver,iname])
		with open(iname,'r') as f:
			l=f.readline()
			a=l.split()
			self.flips=int(a[1]); rcode=int(a[2]); achieved=int(a[10]); minmuls=int(a[11]); plus=int(a[12])
			muls=[]
			for i in range(self.nomuls):
				l=f.readline()
				a=l.split()
				muls.append(int(a[0]))
			fullmuls=[]
			me=[0]*self.nomuls; mf=[0]*self.nomuls
			for i in range(0,self.nomuls,3):
				me[i]=i+2; mf[i]=i+1
				me[i+1]=i; mf[i+1]=i+2
				me[i+2]=i+1; mf[i+2]=i
			for i in range(len(muls)): fullmuls.append([muls[i],muls[me[i]],muls[mf[i]]])
			self.muls=fullmuls
		os.remove(iname)
		tt=time.time()-tt
		if tt>0: spstr=f'{int(60*(self.flips)/tt/1000000)}'
		else: spstr='N/A'
		if ctrls[7]>=2 and plus>0: print('Plus transitions:',plus)
		if rcode==0: st='Target achieved - '
		if rcode==-1:
			if achieved==target: st='Target achieved (zero neighbours) - '
			else: st='State with zero neighbours - '
		if rcode==1: st='Flip limit reached - '
		if rcode==2: st='Terminated early - '
		if rcode==3: st='Weird shit happened - '
		if rcode==4: st='Not implemented - '
		if rcode==5: st='Divergence detected - '
		if rcode==6: st='Escaped infinite loop - '
		if rcode==9: st='No result returned - '
		if rcode==-1 and achieved==target: rcode=0
		self.evalall()
		st+='Flips: '+str(self.flips)+' Speed: '+spstr+' megaflips/min'
		return rcode,minmuls,st

	def evalall(self):
		'''Fully sum all multiplications and calculate current error.'''
		c=self.base
		for mul in self.muls:				
			mc=eval(mul)
			c^=mc
		self.curr=c
		self.err=self.curr.bit_count()

	def writesol(self,fname=None,trans=True):
		'''Write solution to file.'''
		if fname==None: fname='mm21sol.txt'
		if trans: co=1
		else: co=2
		with open(fname,'w') as f:
			for g in range(self.nomuls):
				s='('
				p=False
				en=entriesf(self.muls[g][0])
				for e in en:
					if p: s+='+'
					else: p=True
					s+='a'
					s+=str(row[0][e]+1)
					s+=str(col[0][e]+1)
				s+=')*('
				p=False
				en=entriesf(self.muls[g][1])
				for e in en:
					if p: s+='+'
					else: p=True
					s+='b'
					s+=str(row[1][e]+1)
					s+=str(col[1][e]+1)
				s+=')*('
				p=False
				en=entriesf(self.muls[g][2])
				for e in en:
					if p: s+='+'
					else: p=True
					s+='c'
					s+=str(row[co][e]+1)
					s+=str(col[co][e]+1)
				s+=')\n'
				f.write(s)

	def writecode(self,trans=True):
		'''Write code to implement scheme for further processing.'''
		if trans: co=2
		else: co=1
		print('Code for this scheme:')
		for g in range(self.nomuls):
			s=chr(65+32+g)+'=level3(eladd(['
			p=False
			en=entriesf(self.muls[g][0])
			for e in en:
				if p: s+=','
				else: p=True
				s+='a'
				s+=str(row[0][e]+1)
				s+=str(col[0][e]+1)
			s+=']),eladd(['
			p=False
			en=entriesf(self.muls[g][1])
			for e in en:
				if p: s+=','
				else: p=True
				s+='b'
				s+=str(row[1][e]+1)
				s+=str(col[1][e]+1)
			s+=']))'
			print(s)
		cs=['']*matsize
		for g in range(self.nomuls):
			en=entriesf(self.muls[g][2])
			for e in en: cs[e]+=','+chr(65+32+g)
		for i in range(matsize):
			s='c'
			s+=str(row[co][i]+1)
			s+=str(col[co][i]+1)
			s+='=eladd(['
			s+=cs[i][1:]
			s+='])'
			print(s)
		print()

	def loadsol(self,fname=None,trans=True):
		'''Load solution set from file.'''
		if trans: co=1
		else: co=2
		self.nomuls=0
		self.muls=[]
		with open(fname) as f:
			lines=f.readlines()
			nm=len(lines)
			for l in lines:
				e=[[],[],[]]
				i=0
				star=0
				while i<len(l):
					if l[i]=='*': star+=1
					if l[i]=='a':
						v=(int(l[i+1])-1)*matdim+int(l[i+2])-1
						e[0].append(odr[0][v])
						i+=2
					if l[i]=='b':
						v=(int(l[i+1])-1)*matdim+int(l[i+2])-1
						e[1].append(odr[1][v])
						i+=2						
					if l[i]=='c':
						v=(int(l[i+1])-1)*matdim+int(l[i+2])-1
						e[2].append(odr[co][v])
						i+=2
					i+=1
				vars=[convert(e[0]),convert(e[1]),convert(e[2])]
				self.muls.append(vars)
				self.nomuls+=1
		self.evalall()

	def entrstr(self,n):
		'''Print entries of A,B and C.'''
		lv=len(self.muls[n])
		a=self.muls[n][0]
		if lv==1: b=a; c=a
		else:
			b=self.muls[n][1]
			if lv==2: c=transf(b)
			else: c=self.muls[n][2]
		s='A: |'
		z=bin(a)[2:].zfill(matsize)
		s+=z[::-1]
		s=s+'| B: |'
		z=bin(b)[2:].zfill(matsize)
		s+=z[::-1]
		s=s+'| C: |'
		z=bin(c)[2:].zfill(matsize)
		s+=z[::-1]
		s+='| '
		if lv==1: sz=self.muls[n][0].bit_count()**3
		elif lv==2: sz=self.muls[n][0].bit_count()*self.muls[n][1].bit_count()**2
		else: sz=self.muls[n][0].bit_count()*self.muls[n][1].bit_count()*self.muls[n][2].bit_count()
		s+=f'{n:3}'+f'{sz:4}'
		return(s)

def flipsolver(iname):
	'''Carry out flip graph random walk, written as standalone for porting to C++.'''

	# Load multiplications, and control parameters.
	with open(iname,'r') as f:
		l=f.readline()
		a=l.split()
		nomuls=int(a[0]); flips=int(a[1]); rcode=int(a[2]); target=int(a[3]); flimit=int(a[4])
		plimit=int(a[5]); termination=int(a[6]); rseed=int(a[7]); symm=int(a[8]); maxplus=int(a[9])
		split=int(a[10]); minmuls=int(a[11]); maxsize=int(a[12])
		muls=[]
		for i in range(nomuls):
			l=f.readline()
			a=l.split()
			muls.append(int(a[0]))
	best=copy.copy(muls)

	# Set indices to md,me,mf in muls.
	me=[0]*nomuls; mf=[0]*nomuls
	for i in range(0,nomuls,3):
		me[i]=i+2; mf[i]=i+1
		me[i+1]=i; mf[i+1]=i+2
		me[i+2]=i+1; mf[i+2]=i

	# Set dictionaries for uniques and twoplus.
	uniques={}
	twoplusd={}; twoplusl=[] # Constitute sets with O(1) add, remove, random choice.
	permit=[[1]*nomuls for i in range(nomuls)]
	for i in range(nomuls):
		for j in range(nomuls):
			if i//symm==j//symm: permit[i][j]=0
	achieved=0
	for i in range(nomuls):
		m=muls[i]
		if m>0:
			if m in uniques:
				uniques[m].append(i)
				if m not in twoplusd:
					twoplusd[m]=len(twoplusl)
					twoplusl.append(m)
			else: uniques[m]=[i]
			achieved+=1
	combs=[0,0]; ps=[]; qs=[]
	for x in range(1,80):
		for y in range(x):
			ps.append(x); qs.append(y)
			ps.append(y); qs.append(x)
		combs.append(len(ps))
	plus=0
	rcode=0
	limit=0
	if achieved>=maxplus: plusby=flimit*1007
	elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
	else: plusby=flips+plimit
	limit=updatelimit(limit,flips,termination,split,achieved,target,symm,flimit)
	minmuls=achieved
	nmaxsize=maxsize
	if maxsize<0: nmaxsize=-maxsize
	if ctrls[7]>=3:
		diagnostics(uniques,twoplusl,muls,me,mf,'Status at start:')
		possibilities(uniques,twoplusl,muls,symm,combs,ps,qs)
	if ctrls[17]>0: ctrls[10]=[]

	# Main loop over flips, for 3-way cyclic symmetry.
	if symm==3:
		while True:
			if ctrls[17]>0:
				if flips%ctrls[17]==0: ctrls[10].append(achieved)
			flips+=3

			if maxsize==0: # Set p, q not from same symmetry group, and calculate new values.
				while True:
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					if permit[p][q]: break
				mpe=muls[me[p]]; mpf=muls[mf[p]]
				mqe=muls[me[q]]; mqf=muls[mf[q]]
				mpen=mqe^mpe; mqfn=mqf^mpf

			elif maxsize>0: # Set p, q not from same symmetry group, testing not above certain volume (A*B*C).
				for k in range(1000):
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpen=mqe^mpe; mqfn=mqf^mpf
					psize=muls[p].bit_count()*mpen.bit_count()*mpf.bit_count()
					qsize=muls[q].bit_count()*mqe.bit_count()*mqfn.bit_count()
					if permit[p][q] and psize<=maxsize and qsize<=maxsize: break
				else: rcode=6; break

			else: # Set p, q not from same symmetry group, testing not above certain length (A or B or C).
				for k in range(1000):
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpen=mqe^mpe; mqfn=mqf^mpf
					psize=mpen.bit_count()
					qsize=mqfn.bit_count()
					if permit[p][q] and psize<=nmaxsize and qsize<=nmaxsize: break
				else: rcode=6; break
	
			# Make changes for column e, delete old and add new value.
			flipdel(uniques,twoplusd,twoplusl,me[p],mpe)
			flipadd(uniques,twoplusd,twoplusl,me[p],mpen)
			muls[me[p]]=mpen
	
			# Make changes for column f, delete old and add new value.
			flipdel(uniques,twoplusd,twoplusl,mf[q],mqf)
			flipadd(uniques,twoplusd,twoplusl,mf[q],mqfn)
			muls[mf[q]]=mqfn
	
			# If a row gets a zero in p, remove from all columns, and quit if enough reductions.
			if mpen==0:
				mpd=muls[p]
				flipdel(uniques,twoplusd,twoplusl,p,mpd)
				flipdel(uniques,twoplusd,twoplusl,me[p],mpen)
				flipdel(uniques,twoplusd,twoplusl,mf[p],mpf)
				muls[p]=0; muls[mf[p]]=0
				achieved-=3
				if achieved<minmuls:
					minmuls=achieved
					if achieved>target: limit=updatelimit(limit,flips,termination,split,achieved,target,symm,flimit)
				if achieved<=minmuls: best=copy.copy(muls)
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Reduction:',achieved,flips)
				if len(twoplusl)==0: rcode=-1; break
				if achieved<=target: break
				trigger=True # Check if infinite loop will occur after the zero, if so trigger plus transition.
				for v in twoplusl:
					t=set()
					for w in uniques[v]: t.add(w//3)
					if len(t)>1: trigger=False
				if trigger:
					if ctrls[7]>=3: print('Triggering plus transition to avoid infinite loop:',flips);
					plusby=flips

			# If a row gets a zero in q, remove from all columns, and quit if enough reductions.
			if mqfn==0:
				mqd=muls[q]
				flipdel(uniques,twoplusd,twoplusl,q,mqd)
				flipdel(uniques,twoplusd,twoplusl,me[q],mqe)
				flipdel(uniques,twoplusd,twoplusl,mf[q],mqfn)
				muls[q]=0; muls[me[q]]=0
				achieved-=3
				if achieved<minmuls:
					minmuls=achieved
					if achieved>target: limit=updatelimit(limit,flips,termination,split,achieved,target,symm,flimit)
				if achieved<=minmuls: best=copy.copy(muls)
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Reduction:',achieved,flips)
				if len(twoplusl)==0: rcode=-1; break
				if achieved<=target: break
				trigger=True # Check if infinite loop will occur after the zero, if so trigger plus transition.
				for v in twoplusl:
					t=set()
					for w in uniques[v]: t.add(w//3)
					if len(t)>1: trigger=False
				if trigger:
					if ctrls[7]>=3: print('Triggering plus transition to avoid infinite loop:',flips);
					plusby=flips
	
			# Carry out plus transition if necessary.
			if flips>=plusby:
				for r in range(nomuls):
					if muls[r]==0: break
				while True:
					p=random.randrange(nomuls)
					q=random.randrange(nomuls)
					mpd=muls[p]; mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqd=muls[q]; mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpdn=mpd; mpen=mpe^mqe; mpfn=mpf
					mqdn=mpd; mqen=mqe; mqfn=mpf^mqf
					mrdn=mpd^mqd; mren=mqe; mrfn=mqf
					ok=True
					if maxsize>0:
						psize=mpdn.bit_count()*mpen.bit_count()*mpfn.bit_count()
						qsize=mqdn.bit_count()*mqen.bit_count()*mqfn.bit_count()
						rsize=mrdn.bit_count()*mren.bit_count()*mrfn.bit_count()
						if psize>maxsize or qsize>maxsize or rsize>maxsize: ok=False
					elif maxsize<0:
						psize=mpen.bit_count()
						qsize=mqfn.bit_count()
						rsize=mrdn.bit_count()
						if psize>-maxsize or qsize>-maxsize or rsize>-maxsize: ok=False
					if mpd==0 or mqd==0: ok=False
					if mpd==mqd or mpe==mqe or mpf==mqf: ok=False
					if not permit[p][q]: ok=False
					if ok: break
				flipdel(uniques,twoplusd,twoplusl,me[p],mpe)
				flipadd(uniques,twoplusd,twoplusl,me[p],mpen)
				flipdel(uniques,twoplusd,twoplusl,q,mqd)
				flipadd(uniques,twoplusd,twoplusl,q,mpd)
				flipdel(uniques,twoplusd,twoplusl,mf[q],mqf)
				flipadd(uniques,twoplusd,twoplusl,mf[q],mqfn)
				flipadd(uniques,twoplusd,twoplusl,r,mrdn)
				flipadd(uniques,twoplusd,twoplusl,me[r],mqe)
				flipadd(uniques,twoplusd,twoplusl,mf[r],mqf)
				muls[p]=mpdn; muls[me[p]]=mpen; muls[mf[p]]=mpfn
				muls[q]=mqdn; muls[me[q]]=mqen; muls[mf[q]]=mqfn
				muls[r]=mrdn; muls[me[r]]=mren; muls[mf[r]]=mrfn
				plus+=3
				achieved+=3
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Plus transition:',achieved,flips)

			# Test for termination.
			if flips>=limit:
				if flips>=flimit: rcode=1
				else: rcode=2
				break

	# Main loop over flips, for 6-way cyclic plus reflective symmetry.
	elif symm==6:
		while True:
			if ctrls[17]>0:
				if flips%ctrls[17]==0: ctrls[10].append(achieved)
			flips+=6

			if maxsize==0: # Set p, q not from same symmetry group, and calculate new values.
				while True:
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					if permit[p][q]: break
				mpd=muls[p]; mpe=muls[me[p]]; mpf=muls[mf[p]]
				mqd=muls[q]; mqe=muls[me[q]]; mqf=muls[mf[q]]
				mpen=mqe^mpe; mqfn=mqf^mpf

			elif maxsize>0: # Set p, q not from same symmetry group, testing not above certain volume (A*B*C).
				for k in range(1000):
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					mpd=muls[p]; mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqd=muls[q]; mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpen=mqe^mpe; mqfn=mqf^mpf
					psize=muls[p].bit_count()*mpen.bit_count()*mpf.bit_count()
					qsize=muls[q].bit_count()*mqe.bit_count()*mqfn.bit_count()
					if permit[p][q] and psize<=maxsize and qsize<=maxsize: break
				else: rcode=6; break

			else: # Set p, q not from same symmetry group, testing not above certain length (A or B or C).
				for k in range(1000):
					v=twoplusl[random.randrange(len(twoplusl))]
					x=random.randrange(combs[len(uniques[v])])
					p=uniques[v][ps[x]]
					q=uniques[v][qs[x]]
					mpd=muls[p]; mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqd=muls[q]; mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpen=mqe^mpe; mqfn=mqf^mpf
					psize=mpen.bit_count()
					qsize=mqfn.bit_count()
					if permit[p][q] and psize<=nmaxsize and qsize<=nmaxsize: break
				else: rcode=6; break

			# And for reflective symmetry.
			x=p%6
			if x<3: pp=p+3
			else: pp=p-3
			x=q%6
			if x<3: qq=q+3
			else: qq=q-3
			mppd=muls[pp]; mppe=muls[me[pp]]; mppf=muls[mf[pp]]
			mqqd=muls[qq]; mqqe=muls[me[qq]]; mqqf=muls[mf[qq]]
			mppen=mqqe^mppe; mqqfn=mqqf^mppf

			# Make changes for column e, delete old and add new value.
			flipdel(uniques,twoplusd,twoplusl,me[p],mpe)
			flipadd(uniques,twoplusd,twoplusl,me[p],mpen)
			muls[me[p]]=mpen
			flipdel(uniques,twoplusd,twoplusl,me[pp],mppe)
			flipadd(uniques,twoplusd,twoplusl,me[pp],mppen)
			muls[me[pp]]=mppen
	
			# Make changes for column f, delete old and add new value.
			flipdel(uniques,twoplusd,twoplusl,mf[q],mqf)
			flipadd(uniques,twoplusd,twoplusl,mf[q],mqfn)
			muls[mf[q]]=mqfn
			flipdel(uniques,twoplusd,twoplusl,mf[qq],mqqf)
			flipadd(uniques,twoplusd,twoplusl,mf[qq],mqqfn)
			muls[mf[qq]]=mqqfn
	
			# If a row gets a zero in p, or bonus zero for match with reflection, remove all, and quit if enough reductions.
			if mpen==0 or (mpd==mppd and mpen==mppen and mpf==mppf):
				flipdel(uniques,twoplusd,twoplusl,p,mpd)
				flipdel(uniques,twoplusd,twoplusl,me[p],mpen)
				flipdel(uniques,twoplusd,twoplusl,mf[p],mpf)
				muls[p]=0; muls[mf[p]]=0
				flipdel(uniques,twoplusd,twoplusl,pp,mppd)
				flipdel(uniques,twoplusd,twoplusl,me[pp],mppen)
				flipdel(uniques,twoplusd,twoplusl,mf[pp],mppf)
				muls[pp]=0; muls[mf[pp]]=0
				if mpen!=0: muls[me[p]]=0; muls[me[pp]]=0
				achieved-=6
				if achieved<minmuls:
					minmuls=achieved
					if achieved>target: limit=updatelimit(limit,flips,termination,split,achieved,target,symm,flimit)
				if achieved<=minmuls: best=copy.copy(muls)
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Reduction:',achieved,flips)
				if len(twoplusl)==0: rcode=-1; break
				if achieved<=target: break		
				trigger=True # Check if infinite loop will occur after the zero, if so trigger plus transition.
				for v in twoplusl:
					t=set()
					for w in uniques[v]: t.add(w//6)
					if len(t)>1: trigger=False
				if trigger:
					if ctrls[7]>=3: print('Triggering plus transition to avoid infinite loop:',flips);
					plusby=flips

			# If a row gets a zero in q, or bonus zero for match with reflection, remove all, and quit if enough reductions.
			if mqfn==0 or (mqd==mqqd and mqe==mqqe and mqfn==mqqfn):
				flipdel(uniques,twoplusd,twoplusl,q,mqd)
				flipdel(uniques,twoplusd,twoplusl,me[q],mqe)
				flipdel(uniques,twoplusd,twoplusl,mf[q],mqfn)
				muls[q]=0; muls[me[q]]=0		
				flipdel(uniques,twoplusd,twoplusl,qq,mqqd)
				flipdel(uniques,twoplusd,twoplusl,me[qq],mqqe)
				flipdel(uniques,twoplusd,twoplusl,mf[qq],mqqfn)
				muls[qq]=0; muls[me[qq]]=0
				if mqfn!=0: muls[mf[q]]=0; muls[mf[qq]]=0
				achieved-=6
				if achieved<minmuls:
					minmuls=achieved
					if achieved>target: limit=updatelimit(limit,flips,termination,split,achieved,target,symm,flimit)
				if achieved<=minmuls: best=copy.copy(muls)
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Reduction:',achieved,flips)
				if len(twoplusl)==0: rcode=-1; break
				if achieved<=target: break
				trigger=True # Check if infinite loop will occur after the zero, if so trigger plus transition.
				for v in twoplusl:
					t=set()
					for w in uniques[v]: t.add(w//6)
					if len(t)>1: trigger=False
				if trigger:
					if ctrls[7]>=3: print('Triggering plus transition to avoid infinite loop:',flips);
					plusby=flips
		
			# Carry out plus transition if necessary.
			if flips>=plusby:
				for r in range(nomuls):
					if muls[r]==0: break
				rr=r+3
				while True:
					p=random.randrange(nomuls)
					q=random.randrange(nomuls)
					x=p%6
					if x<3: pp=p+3
					else: pp=p-3
					x=q%6
					if x<3: qq=q+3
					else: qq=q-3
					mpd=muls[p]; mpe=muls[me[p]]; mpf=muls[mf[p]]
					mqd=muls[q]; mqe=muls[me[q]]; mqf=muls[mf[q]]
					mpdn=mpd; mpen=mpe^mqe; mpfn=mpf
					mqdn=mpd; mqen=mqe; mqfn=mpf^mqf
					mrdn=mpd^mqd; mren=mqe; mrfn=mqf
					mppd=muls[pp]; mppe=muls[me[pp]]; mppf=muls[mf[pp]]
					mqqd=muls[qq]; mqqe=muls[me[qq]]; mqqf=muls[mf[qq]]
					mppdn=mppd; mppen=mppe^mqqe; mppfn=mppf
					mqqdn=mppd; mqqen=mqqe; mqqfn=mppf^mqqf
					mrrdn=mppd^mqqd; mrren=mqqe; mrrfn=mqqf
					ok=True
					if maxsize>0:
						psize=mpdn.bit_count()*mpen.bit_count()*mpfn.bit_count()
						qsize=mqdn.bit_count()*mqen.bit_count()*mqfn.bit_count()
						rsize=mrdn.bit_count()*mren.bit_count()*mrfn.bit_count()
						if psize>maxsize or qsize>maxsize or rsize>maxsize: ok=False
					elif maxsize<0:
						psize=mpen.bit_count()
						qsize=mqfn.bit_count()
						rsize=mrdn.bit_count()
						if psize>-maxsize or qsize>-maxsize or rsize>-maxsize: ok=False
					if mpd==0 or mqd==0: ok=False
					if mppd==0 or mqqd==0: ok=False
					if mpd==mqd or mpe==mqe or mpf==mqf: ok=False
					if mppd==mqqd or mppe==mqqe or mppf==mqqf: ok=False
					if not permit[p][q]: ok=False
					if ok: break
				flipdel(uniques,twoplusd,twoplusl,me[p],mpe)
				flipadd(uniques,twoplusd,twoplusl,me[p],mpen)
				flipdel(uniques,twoplusd,twoplusl,q,mqd)
				flipadd(uniques,twoplusd,twoplusl,q,mpd)
				flipdel(uniques,twoplusd,twoplusl,mf[q],mqf)
				flipadd(uniques,twoplusd,twoplusl,mf[q],mqfn)
				flipadd(uniques,twoplusd,twoplusl,r,mrdn)
				flipadd(uniques,twoplusd,twoplusl,me[r],mqe)
				flipadd(uniques,twoplusd,twoplusl,mf[r],mqf)				
				flipdel(uniques,twoplusd,twoplusl,me[pp],mppe)
				flipadd(uniques,twoplusd,twoplusl,me[pp],mppen)
				flipdel(uniques,twoplusd,twoplusl,qq,mqqd)
				flipadd(uniques,twoplusd,twoplusl,qq,mppd)
				flipdel(uniques,twoplusd,twoplusl,mf[qq],mqqf)
				flipadd(uniques,twoplusd,twoplusl,mf[qq],mqqfn)
				flipadd(uniques,twoplusd,twoplusl,rr,mrrdn)
				flipadd(uniques,twoplusd,twoplusl,me[rr],mqqe)
				flipadd(uniques,twoplusd,twoplusl,mf[rr],mqqf)
				muls[p]=mpdn; muls[me[p]]=mpen; muls[mf[p]]=mpfn
				muls[q]=mqdn; muls[me[q]]=mqen; muls[mf[q]]=mqfn
				muls[r]=mrdn; muls[me[r]]=mren; muls[mf[r]]=mrfn
				muls[pp]=mppdn; muls[me[pp]]=mppen; muls[mf[pp]]=mppfn
				muls[qq]=mqqdn; muls[me[qq]]=mqqen; muls[mf[qq]]=mqqfn
				muls[rr]=mrrdn; muls[me[rr]]=mrren; muls[mf[rr]]=mrrfn
				plus+=6
				achieved+=6
				if achieved>=maxplus: plusby=flimit*1007
				elif plimit<0: plusby=flips+random.randrange(-2*plimit)+symm
				else: plusby=flips+plimit
				if ctrls[7]>=3: print('Plus transition:',achieved,flips)

			# Test for termination.
			if flips>=limit:
				if flips>=flimit: rcode=1
				else: rcode=2
				break

	# Write multiplications, and control parameters.
	if ctrls[17]>0: ctrls[10].append(achieved)
	with open(iname,'w') as f:
		s=str(nomuls)+' '+str(flips)+' '+str(rcode)+' '+str(target)+' '+str(flimit)+' '
		s+=str(plimit)+' '+str(termination)+' '+str(rseed)+' '+str(symm)+' '+str(maxplus)+' '
		s+=str(achieved)+' '+str(minmuls)+' '+str(plus)+'\n'
		f.write(s)
		if minmuls<achieved:
			for m in best: s=str(m)+'\n'; f.write(s)
		else:
			for m in muls: s=str(m)+'\n'; f.write(s)

def flipdel(uniques,twoplusd,twoplusl,r,v):
	'''Bookkeeping associated with deleting a multiplication component.'''
	l=len(uniques[v])
	if l==2: 
		rsi=twoplusd[v]; rsl=twoplusl[-1]
		twoplusd[rsl]=rsi; twoplusl[rsi]=rsl
		twoplusl.pop()
		del twoplusd[v]
	if l==1: del uniques[v]
	else: uniques[v].remove(r)

def flipadd(uniques,twoplusd,twoplusl,r,v):
	'''Bookkeeping associated with adding a multiplication component.'''
	if v in uniques:
		l=len(uniques[v])
		uniques[v].append(r)
		if l==1:
			twoplusd[v]=len(twoplusl)
			twoplusl.append(v)
	else: uniques[v]=[r]

def updatelimit(limit,flips,termination,split,achieved,target,symm,flimit):
	'''Update flip limit on obtaining new best rank.'''
	if termination==0: rlimit=flimit
	elif termination==1: steps=(achieved-target)//symm; rlimit=flips+(flimit-flips)//steps
	elif termination==2: rlimit=flips+flimit
	else:
		slimit=split*flimit//100
		if achieved>termination: steps=(achieved-termination)//symm; rlimit=flips+(slimit-flips)//steps
		else: steps=(achieved-target)//symm; rlimit=flips+(flimit-flips)//steps
	return rlimit

def diagnostics(uniques,twoplusl,muls,me,mf,text=None):
	'''Print out state of solver - Python version only.'''
	if text==None: print('Diagnostics:')
	else: print(text)
	print('Uniques:',uniques)
	print('Twoplusl:',len(twoplusl),twoplusl)
	nomuls=len(muls)
	print('Multiplications:')
	for i in range(nomuls):
		print(f'{i:3}',bin(muls[i])[2:].zfill(matsize)[::-1],bin(muls[me[i]])[2:].zfill(matsize)[::-1],bin(muls[mf[i]])[2:].zfill(matsize)[::-1])

def possibilities(uniques,twoplusl,muls,symm,combs,ps,qs):
	'''Print possibilities matrix.'''
	nsm=len(muls)//symm
	poss=[[0]*nsm for i in range(nsm)]
	print('Possibility matrix:')
	for v in twoplusl:
		for x in range(combs[len(uniques[v])]):
			p=uniques[v][ps[x]]
			q=uniques[v][qs[x]]
			pp=p//symm
			qq=q//symm
			if pp!=qq: poss[pp][qq]=1
	for i in range(nsm):
		s='| '
		for j in range(nsm):
			if poss[i][j]==1: s+='* '
			else: s+='. '
		s+='|'
		print(s)

def answer():
	'''Define the answer, flip to changed order, make globally available.'''
	global answ,anslist,altlist
	a=[[[0]*matsize for j in range(matsize)] for k in range(matsize)]
	for k in range(matsize):
		for j in range(matsize):
			for i in range(matsize):
				a[k][j][i]=0
		ci=(k//matdim)*matdim
		cj=k%matdim
		for m in range(matdim):
			ii=ci+m
			jj=cj+matdim*m
			a[k][jj][ii]=1
	flip=[[[0]*matsize for j in range(matsize)] for k in range(matsize)]
	for k in range(matsize):
		for j in range(matsize):
			for i in range(matsize):
				kk=row[2][k]*matdim+col[2][k]
				jj=row[1][j]*matdim+col[1][j]
				ii=row[0][i]*matdim+col[0][i]
				flip[k][j][i]=a[kk][jj][ii]
	anslist=[[a,b,c] for a in range(matsize) for b in range(matsize) for c in range(matsize) if flip[c][b][a]==1]
	mdp=matdim+1
	altlist=[[a,b,c] for a in range(0,matsize,mdp) for b in range(0,matsize,mdp) for c in range(0,matsize,mdp)]
	answ=0
	for an in anslist:
		a=convert([an[0]])
		b=convert([an[1]])
		c=convert([an[2]])
		answ|=eval([a,b,c])
	for ans,alt in zip(anslist,altlist):
		if ans==alt: anslist.remove(ans); altlist.remove(alt)

def entriesf(v):
	'''Returns list of entries from set bits.'''
	return [j for j in range(matsize) if v&1<<j]

def convert(entr):
	'''Convert entries to integer with bits representing them.'''
	a=0
	for e in entr: a|=1<<e
	return a

def eval(x):
	'''Evaluate multiplcation tensor as a long integer.'''
	a,b,c=x
	d=0
	for k in entriesf(b):
		d|=a<<matsize*k
	e=matsize*matsize
	r=0
	for p in entriesf(c): r|=d<<e*p
	return r

def val(e,a,b,c):
	'''Returns value at coordinates a,b,c.'''
	d=a+matsize*b+matsize*matsize*c
	if e&1<<d: return 1
	return 0

def matstr(c):
	'''Return string for print output of a long integer square or cube.'''
	z=bin(c)[2:].zfill(matsize**3)
	zz=[]
	for k in range(matsize): zz.append(z[-matsize*matsize:]); z=z[:-matsize*matsize]
	s='\n'
	for i in range(matsize):
		s1=''
		for j in range(matsize):
			a=0
			f=-1
			for k in range(matsize):
				y=zz[k]
				if int(y[(matsize-i-1)*matsize+(matsize-j-1)])==1:
					a+=1
					if a==1: f=k
			if a==0: s1+='.'
			elif a==1:
				if matdim>5: s1+='1'
				else: s1+=chr(65+f)
			elif a<10: s1+=str(a)
			else: s1+='*'
			s1+=' '
		s+='| '+s1+'|\n'
	return s

def plotres(v,s='',x=None,ylog=False):
	'''Plot result in v.'''
	if ctrls[9]==1: return
	plt.clf()
	if x==None:
		plt.xlim(0,len(v))
		plt.plot(v)	
	else:
		plt.plot(x,v)
	plt.title(s)
	if ylog: plt.yscale('log')
	plt.show()

def plotresm(v,s='',x=None,ylog=False):
	'''Plot many results in v list.'''
	if ctrls[9]==1: return
	plt.clf()
	if x==None:
		plt.xlim(0,len(v[0]))
		for vv in v: plt.plot(vv)	
	else:
		plt.plot(x,v)
	plt.title(s)
	if ylog: plt.yscale('log')
	plt.ylim([10,1000000])
	plt.show()

def plotbar(v,x,s=''):
	'''Plot result in v.'''
	if ctrls[9]==1: return
	plt.clf()
	ymax=max(v)+1
	ymin=0
	plt.bar(x,v,width=1.0)
	plt.title(s)
	plt.ylim([ymin,ymax])
	plt.show()	

def setrco(order):
	'''Set global row, col and ord variables, defining reordering of A,B and C.'''
	if order==0: # Normal row/column order.
		for i in range(3):
			for j in range(matsize): row[i][j]=j//matdim; col[i][j]=j%matdim
	elif order==1: # Ordered with C transposed.
		for i in range(2):
			for j in range(matsize): row[i][j]=j//matdim; col[i][j]=j%matdim
		for j in range(matsize): row[2][j]=col[1][j]; col[2][j]=row[1][j]
	elif order==2: # Ordered by increasing dimension.
		for i in range(3):
			l=0
			for k in range(matdim):
				for j in range(k): row[i][l]=j; col[i][l]=k; l+=1
				for j in range(k): row[i][l]=k; col[i][l]=j; l+=1
				row[i][l]=k; col[i][l]=k; l+=1
	elif order==3: # Ordered by increasing dimension, with C tranposed.
		for i in range(2):
			l=0
			for k in range(matdim):
				for j in range(k): row[i][l]=j; col[i][l]=k; l+=1
				for j in range(k): row[i][l]=k; col[i][l]=j; l+=1
				row[i][l]=k; col[i][l]=k; l+=1
		for j in range(matsize): row[2][j]=col[1][j]; col[2][j]=row[1][j];
	for d in range(3):
		for i in range(matsize):
			r=i//matdim
			c=i%matdim
			for j in range(matsize):
				if row[d][j]==r and col[d][j]==c: odr[d][i]=j

if __name__ == '__main__':
	main()
