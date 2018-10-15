import pickle
import matplotlib.pyplot as plt
import numpy as np

df = pickle.load(open('data.p','rb'))

# Squad = numGroups <= 35
# Duo = 35 < numGroups <= 60
# Solo = 60 < numGroups

solo_df = df[df['numGroups'] > 60]
solo_label = solo_df['winPlacePerc']
solo_data = solo_df.drop(['Id', 'groupId', 'matchId', 'numGroups', 'maxPlace'], axis=1)

pickle.dump(solo_data, open( "solo_data.p", "wb" ))
pickle.dump(solo_label, open( "solo_label.p", "wb" ))
