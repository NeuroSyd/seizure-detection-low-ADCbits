import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample
import stft

from utils.group_seizure_Kaggle import group_seizure

from myio.save_load import save_pickle_file, load_pickle_file, \
    save_hickle_file, load_hickle_file


def load_signals_Kaggle2014Det(data_dir, target, data_type, freq, adcbits):
    print ('load_signals_Kaggle2014Det', target)
    dir = os.path.join(data_dir, target)
    done = False
    i = 0
    while not done:
        i += 1
        filename = '%s/%s_%s_segment_%d.mat' % (dir, target, data_type, i)
        if os.path.exists(filename):
            data = loadmat(filename)
            yield(data)
        else:
            if i == 1:
                raise Exception("file %s not found" % filename)
            done = True

def load_signals_FB(data_dir, target, data_type, adcbits):
    print ('load_signals_FB for Patient', target)

    def strcv(i):
        if i < 10:
            return '000' + str(i)
        elif i < 100:
            return '00' + str(i)
        elif i < 1000:
            return '0' + str(i)
        elif i < 10000:
            return str(i)

    if int(target) < 10:
        strtrg = '00' + str(target)
    elif int(target) < 100:
        strtrg = '0' + str(target)

    if data_type == 'ictal':
        target_ = 'pat%sIktal' % strtrg
        dir = os.path.join(data_dir, target_)
        df_sz = pd.read_csv(
            os.path.join(data_dir,'seizure.csv'),index_col=None,header=0)
        df_sz = df_sz[df_sz.patient==int(target)]
        df_sz.reset_index(inplace=True,drop=True)

        print (df_sz)
        print ('Patient %s has %d seizures' % (target,df_sz.shape[0]))
        for i in range(df_sz.shape[0]):
            data = []
            filename = df_sz.iloc[i]['filename']
            st = df_sz.iloc[i]['start']
            sp = df_sz.iloc[i]['stop']
            print ('Seizure %s starts at %d' % (filename, st))
            for ch in range(1,7):
                filename2 = '%s/%s_%d.asc' % (dir, filename, ch)
                if os.path.exists(filename2):
                    tmp = np.loadtxt(filename2)
                    tmp = tmp[st:sp]
                    tmp = tmp.reshape(tmp.shape[0], 1)
                    data.append(tmp)
                else:
                    raise Exception("file %s not found" % filename)
            if len(data) > 0:
                concat = np.concatenate(data, axis=1)
                print (concat.shape)
                print (np.max(concat), np.min(concat))
                print (concat[::100,0])
                concat = np.round(concat/(2**16)*(2**adcbits))
                print (concat[::100,0])
                yield (concat)

    elif data_type == 'interictal':
        target_ = 'pat%sInteriktal' % strtrg
        dir = os.path.join(data_dir, target_)
        text_files = [f for f in os.listdir(dir) if f.endswith('.asc')]
        prefixes = [text[:8] for text in text_files]
        prefixes = set(prefixes)
        prefixes = sorted(prefixes)

        totalfiles = len(text_files)
        print (prefixes, totalfiles)

        done = False
        count = 0

        for prefix in prefixes:
            i = 0
            while not done:

                i += 1

                stri = strcv(i)
                data = []
                for ch in range(1, 7):
                    filename = '%s/%s_%s_%d.asc' % (dir, prefix, stri, ch)

                    if os.path.exists(filename):
                        try:
                            tmp = np.loadtxt(filename)
                            tmp = tmp.reshape(tmp.shape[0],1)
                            data.append(tmp)
                            count += 1
                        except:
                            print ('OOOPS, this file can not be loaded', filename)
                    elif count >= totalfiles:
                        done = True
                    else:
                        break
                        #raise Exception("file %s not found" % filename)

                if i > 99999:
                    break

                if len(data) > 0:
                    concat = np.concatenate(data, axis=1)
                    print (concat.shape)
                    print (concat[::100,0])
                    concat = np.round(concat/(2**16)*(2**adcbits))
                    print (concat[::100,0])
                    yield (concat)


def load_signals_CHBMIT(data_dir, target, data_type, adcbits):
    print ('load_signals_CHBMIT for Patient', target)
    from mne.io import RawArray, read_raw_edf
    from mne.channels import read_montage
    from mne import create_info, concatenate_raws, pick_types
    from mne.filter import notch_filter

    onset = pd.read_csv(os.path.join(data_dir, 'seizure_summary.csv'),header=0)
    #print (onset)
    osfilenames,sstart,sstop = onset['File_name'],onset['Seizure_start'],onset['Seizure_stop']
    osfilenames = list(osfilenames)
    #print ('Seizure files:', osfilenames)

    segment = pd.read_csv(os.path.join(data_dir, 'segmentation.csv'),header=None)
    nsfilenames = list(segment[segment[1]==0][0])

    nsdict = {
            '0':[]
    }
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
        '23'
    ]
    for t in targets:
        nslist = [elem for elem in nsfilenames if elem.find('chb%s_' %t)!= -1 or elem.find('chb0%s_' %t)!= -1 or elem.find('chb%sa_' %t)!= -1 or elem.find('chb%sb_' %t)!= -1 or elem.find('chb%sc_' %t)!= -1] # could be done much better, I am so lazy
        #nslist = shuffle(nslist, random_state=0)
        nsdict[t] = nslist
    #nsfilenames = shuffle(nsfilenames, random_state=0)

    special_interictal = pd.read_csv(os.path.join(data_dir, 'special_interictal.csv'),header=None)
    sifilenames,sistart,sistop = special_interictal[0],special_interictal[1],special_interictal[2]
    sifilenames = list(sifilenames)

    def strcv(i):
        if i < 10:
            return '0' + str(i)
        elif i < 100:
            return str(i)

    strtrg = 'chb' + strcv(int(target))
    dir = os.path.join(data_dir, strtrg)
    text_files = [f for f in os.listdir(dir) if f.endswith('.edf')]
    #print (target,strtrg)
    print (text_files)

    if data_type == 'ictal':
        filenames = [filename for filename in text_files if filename in osfilenames]
        #print ('ictal files', filenames)
    elif data_type == 'interictal':       
        filenames = text_files
        #print ('interictal files', filenames)

    totalfiles = len(filenames)
    print ('Total %s files %d' % (data_type,totalfiles))
    for filename in filenames:
        exclude_chs = []
        if target in ['4','9']:
            exclude_chs = [u'T8-P8']
        if filename in [
            'chb15_01.edf',   # it has only 18 channels, others have 22
            'chb17c_13.edf',    # only 17 channels, others have 22
            'chb18_01.edf',      # only 17 channels, others have 22
            'chb19_01.edf'
        ]:
            continue

        if target in ['13','16']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'FZ-CZ', u'CZ-PZ']
        elif target in ['4']:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT10-T8']
        else:
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4', u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ', u'P7-T7', u'T7-FT9', u'FT9-FT10', u'FT10-T8']


        rawEEG = read_raw_edf('%s/%s' % (dir, filename),
                              #exclude=exclude_chs,  #only work in mne 0.16
                              verbose=0,preload=True)
        rawEEG.pick_channels(chs)
        print (filename,rawEEG.ch_names)
        #rawEEG.notch_filter(freqs=np.arange(60,121,60))



        tmp = rawEEG.to_data_frame()
        tmp = tmp.values
        #print (tmp.shape)
        #ica = FastICA(n_components=data.shape[1])
        #data = ica.fit_transform(data)  # Reconstruct signals
        if data_type == 'ictal':
            # get onset information
            indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
            if len(indices) > 0: #multiple seizures in one file
                print ('%d seizures in the file %s' % (len(indices),filename))
                for i in range(len(indices)):
                    st = sstart[indices[i]]*256
                    sp = sstop[indices[i]]*256
                    print ('%s: Seizure %d starts at %d stops at %d' % (filename, i, st,sp))
                    data = tmp[st:sp]
                    print ('data shape', data.shape)
                    print ('data shape', data.shape)
                    print (np.max(data), np.min(data))
                    # print (data[::100,0])
                    if adcbits < 16:
                        data = np.round(data*10.2375/(2**16)*(2**adcbits))
                    # 10.2375 is the scaling factor of ADC
                    # print (data[::100,0])
                    yield(data)


        elif data_type == 'interictal':
            if filename in osfilenames:
                # get onset information
                indices = [ind for ind,x in enumerate(osfilenames) if x==filename]
                data_ = []
                if len(indices) == 1: # only one seizure
                    st = sstart[indices[0]]*256
                    sp = sstop[indices[0]]*256
                    tmp1 = tmp[:st]
                    tmp2 = tmp[sp+256:]
                    data_.append(tmp1)
                    data_.append(tmp2)
                    print ('DEBUG: 1 SZ', tmp1.shape, tmp2.shape)
                elif len(indices) == 2: # two seizures in one file
                    print ('%d seizures in the file %s' % (len(indices),filename))
                    st1 = sstart[indices[0]]*256
                    sp1 = sstop[indices[0]]*256
                    st2 = sstart[indices[1]]*256
                    sp2 = sstop[indices[1]]*256
                    tmp1 = tmp[:st1]
                    tmp2 = tmp[sp1+256:st2]
                    tmp3 = tmp[sp2+256:]
                    data_.append(tmp1)
                    data_.append(tmp2)
                    data_.append(tmp3)
                    print ('DEBUG: 2 SZ', tmp1.shape, tmp2.shape, tmp3.shape)
                elif len(indices) > 2: # more than two seizures
                    st0 = sstart[indices[0]]*256
                    spn = sstop[indices[-1]]*256
                    tmp0 = tmp[:st0]
                    tmpn = tmp[spn+256:]
                    data_.append(tmp0)
                    for i in range(1,len(indices)-1):
                        sp_p = sstop[indices[i-1]]*256 # end of previous seizure
                        st = sstart[indices[i]]*256
                        sp = sstop[indices[i]]*256
                        st_n = sstart[indices[i+1]]*256 # start of next seizure

                        tmp1 = tmp[sp_p+256:st]
                        tmp2 = tmp[sp+256:st_n]
                        data_.append(tmp1)
                        data_.append(tmp2)
                    data_.append(tmpn)
                    print ('DEBUG: >2 SZ', tmp0.shape, tmpn.shape, len(data_))
                data = np.concatenate(data_, axis=0)

            else:
                data = tmp
            print ('data shape', data.shape)
            # print (data[::100,0])
            if adcbits < 16:
                data = np.round(data*10.2375/(2**16)*(2**adcbits))
            # 10.2375 is the scaling factor of ADC
            # print (data[::100,0])
            yield(data)

class PrepData():
    def __init__(self, target, type, settings, adcbits):
        self.target = target
        self.settings = settings
        self.type = type
        self.adcbits = adcbits


    def read_raw_signal(self):
        if self.settings['dataset'] in ['CHBMIT']:
            self.freq = 256            
            return load_signals_CHBMIT(
                self.settings['datadir'], self.target, self.type, self.adcbits)
        elif self.settings['dataset'] == 'FB':
            self.freq = 256            
            return load_signals_FB(
                self.settings['datadir'], self.target, self.type, self.adcbits)
        elif self.settings['dataset'] == 'Kaggle2014Det':
            self.freq = 400
            self.significant_channels = None
            return load_signals_Kaggle2014Det(
                self.settings['datadir'], self.target, self.type, self.freq, self.adcbits)

        return 'array, freq, misc'

    def preprocess(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq
        numts = 1
        
        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0,index_col=None)
        trg = int(self.target)
        print (df_sampling)
        print (df_sampling[df_sampling.Subject==trg].ictal_ovl.values)
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==trg].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)

        def process_raw_data(mat_data):            
            print ('Loading data')
            #print mat_data
            X = []
            y = []
    
            #scale_ = scale_coef[target]
    
            for data in mat_data:
                if ictal:
                    y_value=1
                    first_segment = False
                else:
                    y_value=0
    
                X_temp = []
                y_temp = []
    
                totalSample = int(data.shape[0]/targetFrequency/numts) + 1
                window_len = int(targetFrequency*numts)
                for i in range(totalSample):
                    if (i+1)*window_len <= data.shape[0]:
                        s = data[i*window_len:(i+1)*window_len,:]
                        # p90 = np.percentile(np.abs(s),90)
                        # # print ('1. p90', p90)
                        # if p90 < 10.0*(2**self.adcbits)/(2**16):
                        #     continue
    
                        stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                        stft_data = np.abs(stft_data)+1e-6
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        stft_data = np.transpose(stft_data, (1, 2, 0))
                        if self.settings['dataset'] == 'FB':
                            stft_data = np.concatenate((stft_data[:,:,1:47],
                                                        stft_data[:,:,54:97],
                                                        stft_data[:,:,104:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'CHBMIT':
                            stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,64:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)  
    
                        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])    
    
                        X_temp.append(stft_data)
                        y_temp.append(y_value)
    
                #overlapped window
                if ictal:
                    i = 1
                    print ('ictal_ovl_len =', ictal_ovl_len)
                    while (window_len + (i + 1)*ictal_ovl_len <= data.shape[0]):
                        s = data[i*ictal_ovl_len:i*ictal_ovl_len + window_len, :]
                        p90 = np.percentile(np.abs(s),90)
                        # print ('2. p90', p90)
                        # if p90 < 10.0*(2**self.adcbits)/(2**16):
                        #     i += 1
                        #     continue
    
                        stft_data = stft.spectrogram(s, framelength=targetFrequency,centered=False)
                        stft_data = np.abs(stft_data)+1e-6
                        stft_data = np.log10(stft_data)
                        indices = np.where(stft_data <= 0)
                        stft_data[indices] = 0

                        stft_data = np.transpose(stft_data, (1, 2, 0))
                        if self.settings['dataset'] == 'FB':
                            stft_data = np.concatenate((stft_data[:,:,1:47],
                                                        stft_data[:,:,54:97],
                                                        stft_data[:,:,104:]),
                                                       axis=-1)
                        elif self.settings['dataset'] == 'CHBMIT':
                            stft_data = np.concatenate((stft_data[:,:,1:57],
                                                        stft_data[:,:,64:117],
                                                        stft_data[:,:,124:]),
                                                       axis=-1)    
                       
                        stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                      stft_data.shape[1],
                                                      stft_data.shape[2])                       
    
                        X_temp.append(stft_data)
                        # to differentiate between non overlapped and overlapped
                        # samples. Testing only uses non overlapped ones.
                        y_temp.append(2)
                        i += 1
    
    
                X_temp = np.concatenate(X_temp, axis=0)
                y_temp = np.array(y_temp)
                X.append(X_temp)
                y.append(y_temp)
    

            if ictal or interictal:
                #y = np.array(y)
                print ('X', len(X), X[0].shape, 'y', len(y), y[0].shape)
                return X, y
            else:
                print ('X', X.shape)
                return X

        data = process_raw_data(data_)

        return  data

    def preprocess_Kaggle2014Det(self, data_):
        ictal = self.type == 'ictal'
        interictal = self.type == 'interictal'
        targetFrequency = self.freq   #re-sample to target frequency
    
        df_sampling = pd.read_csv(
            'sampling_%s.csv' % self.settings['dataset'],
            header=0,index_col=None)
        print (df_sampling)
        print (df_sampling[df_sampling.Subject==self.target].ictal_ovl.values)
        
        ictal_ovl_pt = \
            df_sampling[df_sampling.Subject==self.target].ictal_ovl.values[0]
        ictal_ovl_len = int(targetFrequency*ictal_ovl_pt)

        interictal_ovl_pt = \
            df_sampling[df_sampling.Subject==self.target].interictal_ovl.values[0]
        interictal_ovl_len = int(targetFrequency*interictal_ovl_pt)


        # for each data point in ictal, interictal and test,
        # generate (X, <y>, <latency>) per channel
        def process_raw_data(mat_data, gen_ictal, gen_interictal):

            print ('Loading data',self.type)
            X = []
            y = []
            X_temp = []
            y_temp = []
            latencies = []
            latency = -1
            prev_latency = -1
            prev_data = None
    
            for segment in mat_data:
                data = segment['data']
                if (self.significant_channels is not None):
                    data = data[self.significant_channels]
                if data.shape[-1] > targetFrequency:
                    data = resample(data, targetFrequency, axis=data.ndim - 1)
    
                data = np.transpose(data)
                if ictal:
                    if gen_ictal and (prev_data is not None):
                        i_gen=1
                        while (i_gen*ictal_ovl_len<data.shape[0]):
                            a = prev_data[i_gen*ictal_ovl_len:,:]
                            b = data[:i_gen*ictal_ovl_len,:]
                            #print ('a b shapes', a.shape, b.shape)
                            #c = np.concatenate((a,b),axis=1)
                            #print ('c shape', c.shape)
                            gen_data = np.concatenate((a,b),axis=0)
                            i_gen += 1

                            stft_data = stft.spectrogram(
                                gen_data, framelength=targetFrequency,centered=False)
                            stft_data = np.abs(stft_data)+1e-6
                            stft_data = np.log10(stft_data)
                            indices = np.where(stft_data <= 0)
                            stft_data[indices] = 0

                            stft_data = np.transpose(stft_data, (1, 2, 0))

                            stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                          stft_data.shape[1],
                                                          stft_data.shape[2])

                            X_temp.append(stft_data)
                            if prev_latency <= 15:
                                y_value = 2 # ictal <= 15
                            else:
                                y_value = 2 # ictal > 15
                            y_temp.append(y_value)
                            latencies.append(prev_latency)
    
                elif interictal:
                    if gen_interictal and (prev_data is not None):
                        i_gen=1
                        while (i_gen*interictal_ovl_len<data.shape[0]):
                            a = prev_data[i_gen*interictal_ovl_len:,:]
                            b = data[:i_gen*interictal_ovl_len,:]
                            gen_data = np.concatenate((a,b),axis=0)
                            i_gen += 1
                            stft_data = stft.spectrogram(
                                gen_data, framelength=targetFrequency,centered=False)
                            stft_data = np.abs(stft_data)+1e-6
                            stft_data = np.log10(stft_data)
                            indices = np.where(stft_data <= 0)
                            stft_data[indices] = 0

                            stft_data = np.transpose(stft_data, (1, 2, 0))
                          
                            stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                                          stft_data.shape[1],
                                                          stft_data.shape[2])
                            X.append(stft_data)
                            y.append(-1)
    
                    y.append(0)
    


                stft_data = stft.spectrogram(
                    data, framelength=targetFrequency,centered=False)
                stft_data = np.abs(stft_data)+1e-6
                stft_data = np.log10(stft_data)
                indices = np.where(stft_data <= 0)
                stft_data[indices] = 0
                stft_data = np.transpose(stft_data, (1, 2, 0))               

                stft_data = stft_data.reshape(-1, stft_data.shape[0],
                                              stft_data.shape[1],
                                              stft_data.shape[2])
                if ictal:
                    latency = segment['latency'][0]
                    if latency <= 15:
                        y_value = 1 # ictal <= 15
                    else:
                        y_value = 1 # ictal > 15
                    X_temp.append(stft_data)
                    y_temp.append(y_value)
                    latencies.append(latency)
                else:
                    X.append(stft_data)

                prev_data = data
                prev_latency = latency

    
            #X = np.array(X)
            #y = np.array(y)
    
            if ictal:
                X, y = group_seizure(X_temp, y_temp, latencies)
                print ('Number of seizures %d' %len(X), X[0].shape, y[0].shape)
                return X, y
            elif interictal:
                X = np.concatenate(X)
                y = np.array(y)
                print ('X', X.shape, 'y', y.shape)
                return X, y
            else:
                X = np.concatenate(X)
                print ('X test', X.shape)
                return X, None
    
        data = process_raw_data(data_, gen_ictal=True, gen_interictal=True)
    
        return data

    def apply(self):
        filename = '%s_%s' % (self.type, self.target)
        cache = load_hickle_file(
            os.path.join(self.settings['cachedir'], filename))
        if cache is not None:
            return cache

        data = self.read_raw_signal()
        if self.settings['dataset'] in ['CHBMIT','FB']:
            X, y = self.preprocess(data)
        elif self.settings['dataset'] in ['Kaggle2014Det']:
            X, y = self.preprocess_Kaggle2014Det(data)
        save_hickle_file(
            os.path.join(self.settings['cachedir'], filename),
            [X, y])
        return X, y


