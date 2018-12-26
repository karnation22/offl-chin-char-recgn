# -*- coding: utf-8 -*-
from __future__ import unicode_literals

JOIN = lambda base,ext: base+"\\"+ext
STR_WRITE = lambda chin_index,NP2D: (str(chin_index)+" "+"".join([str(feat_index+1)+":"+str(round(float(val)/256, 4))+" " 
			for feat_index,val in enumerate(list(NP2D))])+"\n")
