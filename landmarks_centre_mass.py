import numpy as np
import nibabel as nib
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--inputnii", dest="inputnii", help="input segmentation from (nii.gz)", default=None, required=True)
    parser.add_argument("--movingnii", dest="movingnii",  help="second segmentation from (nii.gz)", default=None, required=False)
    parser.add_argument("--savetxt", dest="savetxt",  help="output landmark file to (txt)", default=None, required=False)

    options = parser.parse_args()
    d_options = vars(options)
   
    img = nib.load(d_options['inputnii'])
    img_data = img.get_data()
    x = np.linspace(0, img_data.shape[0]-1, img_data.shape[0])
    y = np.linspace(0, img_data.shape[1]-1, img_data.shape[1])
    z = np.linspace(0, img_data.shape[2]-1, img_data.shape[2])
    yv, xv, zv = np.meshgrid(y,x,z)
    unique = np.unique(img_data)
    positions = np.zeros((len(unique)-1,3))
    for i in range(1,len(unique)):        
        label = (img_data==unique[i]).astype('float32')
        xc = np.sum(label*xv)/np.sum(label)
        yc = np.sum(label*yv)/np.sum(label)
        zc = np.sum(label*zv)/np.sum(label)
        positions[i-1,0] = xc
        positions[i-1,1] = yc
        positions[i-1,2] = zc
        if(d_options['savetxt'] is None):
            print(('label',unique[i],'x',xc,'y',yc,'z',zc))
        
    if(d_options['movingnii'] is not None):
        img2 = nib.load(d_options['movingnii'])
        img_data2 = img2.get_data()
        positions2 = np.zeros((len(unique)-1,3))

        for i in range(1,len(unique)):        
            label = (img_data2==unique[i]).astype('float32')
            xc = np.sum(label*xv)/np.sum(label)
            yc = np.sum(label*yv)/np.sum(label)
            zc = np.sum(label*zv)/np.sum(label)
            positions2[i-1,0] = xc
            positions2[i-1,1] = yc
            positions2[i-1,2] = zc
            if(d_options['savetxt'] is None):
                print(('label2',unique[i],'x',xc,'y',yc,'z',zc))

        error = np.mean(np.sqrt(np.sum(np.power(positions-positions2,2),1)))
        print(('landmark error (vox)',error))
        
    with open(d_options['savetxt']+"_mri.txt", "w") as text_file:
        for i in range(positions.shape[0]):
            text_file.write("%f %f %f %d \n" % (positions[i,0],positions[i,1],positions[i,2],i))   
    with open(d_options['savetxt']+"_us.txt", "w") as text_file:
        for i in range(positions2.shape[0]):
            text_file.write("%f %f %f %d \n" % (positions2[i,0],positions2[i,1],positions2[i,2],i)) 
    

if __name__ == '__main__':
    main()
