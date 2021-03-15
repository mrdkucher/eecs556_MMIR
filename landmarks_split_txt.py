import numpy as np
import argparse

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--inputtag", dest="inputtag", help="input tag file from (MINC)", default=None, required=True)
    parser.add_argument("--savetxt", dest="savetxt",  help="output landmark file to (txt)", default=None, required=True)


    options = parser.parse_args()
    d_options = vars(options)
   
    landmarks = []
    with open(d_options['inputtag']) as f:
        lines=f.readlines()
        for line in lines:
            myarray = np.fromstring(line, dtype=float, sep=' ')
            if(len(myarray)==6):
                landmarks.append(myarray)

    landmarks = np.asarray(landmarks)
    #print(landmarks)
    
    with open(d_options['savetxt']+"_mri.txt", "w") as text_file:
        for i in range(landmarks.shape[0]):
            text_file.write("%f %f %f %d \n" % (landmarks[i,0],landmarks[i,1],landmarks[i,2],i))   
    with open(d_options['savetxt']+"_us.txt", "w") as text_file:
        for i in range(landmarks.shape[0]):
            text_file.write("%f %f %f %d \n" % (landmarks[i,3],landmarks[i,4],landmarks[i,5],i))   
    
    
if __name__ == '__main__':
    main()
