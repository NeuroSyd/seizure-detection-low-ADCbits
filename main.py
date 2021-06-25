import json
import os
import os.path
import numpy as np
import pandas as pd

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import keras
keras.backend.set_image_data_format('channels_first')
print ('Using Keras image_data_format=%s' % keras.backend.image_data_format())

from utils.load_signals import PrepData
from utils.prep_data import train_val_loo_split, train_val_test_split, train_val_split
from myio.save_load import write_out
from utils.log import log
from models.cnn import ConvNN

def makedirs(dir):
    try:
        os.makedirs(dir)
    except:
        pass


def main(dataset, build_type, adcbits):
    print ('Main')
    with open('SETTINGS_%s.json' %dataset) as f:
        settings = json.load(f)
    makedirs(str(settings['cachedir']))
    makedirs(str(settings['resultdir']))

    if settings['dataset'] == 'CHBMIT':
        # skip Patient 12, not able to read channels name in chb12_09.edf
        targets = [
            '1',
            # '2',
            # '3',
            # '4',
            # '5',
            # '6',
            # '7',
            # '8',
            # '9',
            # '10',
            # '11',
           
            # '13',
            # '14',
            # '15',
            # '16',
            # '17',
            # '18',
            # '19',
            # '20',
            # '21',
            # '22',
            # '23'
        ]
    elif settings['dataset'] == 'FB':
        targets = [
            '1',
            '3',
            '4',
            '5',
            '6',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
        ]
    elif settings['dataset'] == 'Kaggle2014Det':
        targets = [
            'Dog_1',
            'Dog_2',
            'Dog_3',
            'Dog_4',
            'Patient_1',
            'Patient_2',
            'Patient_3',
            'Patient_4',
            'Patient_5',
            'Patient_6',
            'Patient_7',
            'Patient_8',

        ]
    elif settings['dataset']=='EpilepsiaSurf':
        targets = [
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
            '10',
            '11',
            '12',
            '13',
            '14',
            '15',
            '16',
            '17',
            '18',
            '19',
            '20',
            '21',
            '22',
            '23',
            '24',
            '25',
            '26',
            '27',
            '28',
            '29',
            '30'
        ]

    summary = {}
    lines = ['clip,seizure']
    for target in targets:
        count = 1
        ictal_X, ictal_y = \
            PrepData(target, type='ictal', settings=settings, adcbits=adcbits).apply()
        interictal_X, interictal_y = \
            PrepData(target, type='interictal', settings=settings, adcbits=adcbits).apply()

        if build_type=='cv':
            loo_folds = train_val_loo_split(
                ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
            ind = 1
            for X_train, y_train, X_val, y_val, X_test, y_test in loo_folds:
                print (X_train.shape, y_train.shape,
                       X_val.shape, y_val.shape,
                       X_test.shape, y_test.shape)
                print ('y values', np.unique(y_train), np.unique(y_val), np.unique(y_test))
                
                model = ConvNN(
                    target,batch_size=32,nb_classes=2,epochs=50,mode=build_type)
               
                model.setup(X_train.shape)
                model.fit(X_train, y_train, X_val, y_val)
                auc = model.evaluate(X_test, y_test)
                t = '%s_%d' %(target, ind)
                summary[t] = auc
				
                # write out predictions for preictal and interictal segments
                # preictal
                X_test_p = X_test[y_test==1]
                y_test_p = model.predict_proba(X_test_p)
                filename = os.path.join(
                    str(settings['resultdir']),
                    'preictal_%s_%d.csv' %(target, ind))
                write_out(y_test_p, filename)

                X_test_i = X_test[y_test==0]
                y_test_i = model.predict_proba(X_test_i)
                filename = os.path.join(
                    str(settings['resultdir']),
                    'interictal_%s_%d.csv' %(target, ind))
                write_out(y_test_i, filename)
					
                ind += 1
				
        elif build_type=='test': # apply for Kaggle2014Det dataset only
            X_test, _ = \
            PrepData(target, type='test', settings=settings).apply()
            X_train, y_train, X_val, y_val = train_val_split(
                ictal_X, ictal_y, interictal_X, interictal_y, 0.25)
            df_key = pd.read_csv(
                'seizureDetectionAnswerKey.csv',
                header=0,index_col=None)
            print (df_key.head())
            print (df_key[df_key['clip'].str.contains(target)].seizure.values)
            y_test = df_key[df_key['clip'].str.contains(target)].seizure.values
            y_test = np.array(y_test)
            
            model = ConvNN(
                target,batch_size=32,nb_classes=2,epochs=50,mode=build_type)
            
            model.setup(X_train.shape)
            #model.fit(X_train, y_train)
            fn_weights = "weights_%s_%s.h5" %(target, build_type)
            if os.path.exists(fn_weights):
                model.load_trained_weights(fn_weights)
            else:
                model.fit(X_train, y_train, X_val, y_val)
            auc = model.evaluate(X_test, y_test)
            summary[target] = auc

            # write out predictions for preictal and interictal segments
            y_test_p = model.predict_proba(X_test)
            for p in range(y_test_p.shape[0]):
                lines.append('%s_test_segment_%d,%.4f' %(target,count,y_test_p[p,1]))
                count += 1

    print (summary)
    log(str(summary))
   

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="FB",
                        help="FB, CHBMIT or Kaggle2014Det")
    parser.add_argument("--mode", default="cv",
                        help="cv or test. cv is for leave-one-out cross-validation")   
    
    parser.add_argument("--adcbits", type=int, default=16,
                        help="specify number of bits of the ADC")
    args = parser.parse_args()
    assert args.dataset in ["FB", "CHBMIT", "Kaggle2014Det"]
    assert args.mode in ['cv','test']
   
    log('********************************************************************')
    log('--- START --dataset %s --mode %s --adcbits %s ---'
        %(args.dataset,args.mode,args.adcbits))
    main(
        dataset=args.dataset,
        build_type=args.mode,               
        adcbits=args.adcbits
    )
