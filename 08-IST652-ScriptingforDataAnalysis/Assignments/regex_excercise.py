# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 09:50:25 2019

@author: rkrishnan
"""
import re

text = '''Celebrate #NationalPetDay with our puppy playlist: https://t.co/eBHHFPW0z7 https://t.co/uix5AY2FFQ<a href="http://msande.stanford.edu"> Management Science and Engineering </a><p

      Address: Terman 311, Stanford CA 94305<br>

      Email: ashishg@cs.stanford.edu<br>

      Phone: (650)814-9999 [Cell], Fax: (650)723-9999<br>

      Admin asst: Roz Morf, Terman 405, 650-723-9999, rozm@stanford.edu</p>

The U.S.A. olympic teams have east-west training centers with up-to-date equipment.

'''

#Match URLs: https?:\/\/[a-zA-Z]+.\w+\/?\.?\w+
#Match Phone Numbers: [(]?\d{3}-?\)?\d{3}-?\d{4}
#Match Email Address: \w+@[a-zA-Z.]+
#Match Acronyms: [A-Z][A-Z.]+(\s)
#Match word with hypens: [a-zA-Z]+-[a-zA-Z]+-?[a-zA-Z]+

#Match URLs

pattern=re.compile("https?:\/\/[a-zA-Z]+.\w+\/?\.?\w+")
matches=pattern.findall(text)

for match in matches:
    print(match)

#https://t.co/eBHHFPW0z7
#https://t.co/uix5AY2FFQ
#http://msande.stanford.edu
    
    
#Match Phone Number

pattern=re.compile("[(]?\d{3}-?\)?\d{3}-?\d{4}")
matches=pattern.findall(text)

for match in matches:
    print(match)

#(650)814-9999
#(650)723-9999
#650-723-9999
#Match Email Address: \w+@[a-zA-Z.]+
    
pattern=re.compile("\w+@[a-zA-Z.]+")
matches=pattern.findall(text)

for match in matches:
    print(match)
    
#ashishg@cs.stanford.edu
#rozm@stanford.edu
#Match Acronyms: [A-Z][A-Z.]+(\s)
    
pattern=re.compile("[A-Z][A-Z.]+\s")
matches=pattern.findall(text)

for match in matches:
    print(match)
    
#CA 
#U.S.A. 
#Match word with hypens: [a-zA-Z]+-[a-zA-Z]+-?[a-zA-Z]+
    
pattern=re.compile("[a-zA-Z]+-[a-zA-Z]+-?[a-zA-Z]+")
matches=pattern.findall(text)

for match in matches:
    print(match)

#east-west
#up-to-date