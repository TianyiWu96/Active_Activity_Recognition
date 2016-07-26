
from pandas import Series

def Loading_HAPT(foldername,data):
    labelfile=foldername+'labels.txt'
    with open(labelfile) as f:
    	for line in f:
    		tokens = line.split()
    		experiment_id = tokens[0]
    		user_id = tokens[1]
    		activity = int(tokens[2])
    		start = int(tokens[3])-1
    		end = int(tokens[4])-1
    		if(int(experiment_id)<10):
    			experiment_id ='0'+experiment_id
    			# print(experiment_id)
    		if(int(user_id)<10):
    			user_id='0'+user_id
    		with open(foldername+'acc_exp'+experiment_id+'_user'+user_id+'.txt') as m:
    			lines = m.readlines()
    			for i in range(start,end+1):
    				raw = lines[i].split()
    				# print(data)
    				data['x'].append(float(raw[0]))
    				data['y'].append(float(raw[1]))
    				data['z'].append(float(raw[2]))
    				data['User'].append(int(user_id))
    				data['activity'].append(int(activity))
    				data['timestamp'].append(i+1)
    return data









