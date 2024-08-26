import os
import io

dir = 'data/lerf_ovs/waldo_kitchen/language_features_dim3'
dist = 'data/lerf_ovs/waldo_kitchen/temp'
files=os.listdir(dir)
sfiles = list(filter(lambda x: x.endswith('s.npy'), files))
ffiles = list(filter(lambda x: x.endswith('f.npy'), files))
sfiles.sort()
ffiles.sort()
for i,file in enumerate(sfiles):
    print(file, file.replace(str(i).rjust(5,'0'), str(i+1).rjust(5,'0')))
    with open(os.path.join(dir, file), 'rb') as f:
        if i<154:
            with open(os.path.join(dist, file.replace(str(i).rjust(5,'0'), str(i+1).rjust(5,'0'))), 'wb') as df:
                df.write(f.read())
        else:
            with open(os.path.join(dist, file.replace(str(i).rjust(5,'0'), str(i+4).rjust(5,'0'))), 'wb') as df:
                df.write(f.read())
for i,file in enumerate(ffiles):
    print(file, file.replace(str(i).rjust(5,'0'), str(i+1).rjust(5,'0')))
    with open(os.path.join(dir, file), 'rb') as f:
        if i<154:
            with open(os.path.join(dist, file.replace(str(i).rjust(5,'0'), str(i+1).rjust(5,'0'))), 'wb') as df:
                df.write(f.read())
        else:
            with open(os.path.join(dist, file.replace(str(i).rjust(5,'0'), str(i+4).rjust(5,'0'))), 'wb') as df:
                df.write(f.read())
