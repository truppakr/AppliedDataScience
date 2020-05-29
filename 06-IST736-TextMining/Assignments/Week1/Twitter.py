# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 07:46:50 2019

@author: rkrishnan
"""

import json

# Enter your keys/secrets as strings in the following fields
credentials = {}  
credentials['CONSUMER_KEY'] = "bSg7qo6oMrRDSuUGblRVUODQZ"
credentials['CONSUMER_SECRET'] = "7Coc2qcwMViDmV3Ye5YGMIloLX2pzQkopK4AWhlg0lZbUXXCSd"
credentials['ACCESS_TOKEN'] = "1938157304-VTcQXvcuQ1Hokinv0tlaqCwnxzp9np1fIaqxGS8"
credentials['ACCESS_SECRET'] = "eHbSPVkNdcLiMvJfxWTuZ9y6vMLJmIdwIU5qqszL4Ifju"

# Save the credentials object to file
with open("twitter_credentials.json", "w") as file:  
    json.dump(credentials, file)