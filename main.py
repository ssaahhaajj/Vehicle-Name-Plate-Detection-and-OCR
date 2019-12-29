import SEGMENTATION
import pickle
print("Prediting the Liccence Plate")
filename = './model.sav'
model = pickle.load(open(filename, 'rb'))

res = []
for ch in SEGMENTATION.Rec_char:
    ch = ch.reshape(1, -1);
    result = model.predict(ch)
    res.append(result)
    
plate_string = ''
for ch in res:
    plate_string += ch[0]

new_col_lis = SEGMENTATION.col_lis[:]
SEGMENTATION.col_lis.sort()
rightplate_string = ''
for each in SEGEMENTATION.col_lis:
    rightplate_string += plate_string[new_col_lis.index(each)]

#detecting if the IND is also detected
if rightplate_string[:3]=='IND':
    rightplate_string = rightplate_string[3:]
    
print('Predicted License plate is:')
print(rightplate_string)
