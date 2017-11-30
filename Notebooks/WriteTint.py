import json, datetime, os, struct
import numpy as np


class WriteTint:

    def write_set(self, fname, active_chan, Fs, time, create_date, create_time):

        expter_file = 'experimenter.json'
        with open(expter_file, 'r') as f:
            expter_dict = json.load(f)

        for key, value in expter_dict.items():
            if self.exptername.currentText() == key:
                expter_values = value
            else:
                pass
        for i in range(0, len(expter_values)):
            if value[i] == '':
                if i == 1:  # min x
                    expter_values[i] = '0'
                elif i == 2:  # max x
                    expter_values[i] = '768'
                elif i == 3:  # min y
                    expter_values[i] = '0'
                elif i == 4:  # max y
                    expter_values[i] = '574'
                elif i == 5:  # min x window
                    expter_values[i] = '233'
                elif i == 6:  # max x window
                    expter_values[i] = '439'
                elif i == 7:  # min y window
                    expter_values[i] = '144'
                elif i == 8:
                    expter_values[i] = '350'
                elif i == 9:
                    expter_values[i] = '700'  # pix/meter

        with open(fname, 'w') as f:
            date = 'trial_date %s' % (create_date)
            time_head = '\ntrial_time %s' % (create_time)
            expter = '\nexperimenter %s' % (self.exptername.currentText())
            comments = '\ncomments %s' % (self.comment_e.toPlainText())
            # dur = '\nduration %d' % (int(math.ceil(time[-1])))
            dur = '\nduration %.2f' % (time[-1])
            sw_vers = '\nsw_version %s' % (self.sw_combo.currentText())
            order = [date, time_head, expter, comments, dur, sw_vers]

            f.seek(0, 2)
            f.writelines(order)

            lines = ['ADC_fullscale_mv 1500',
                     'tracker_version 0',
                     'stim_version 1',
                     'audio_version 0']

            f.seek(0, 2)
            f.writelines(lines)

            for chan in range(0, len(active_chan)):
                lines = ['\ngain_ch_%d 20000' % (chan),
                         '\nfilter_ch_%d 2' % (chan),
                         '\na_in_ch_%d %d' % (chan, chan),
                         '\nb_in_ch_%d 11' % (chan),
                         '\nmode_ch_%d 5' % (chan),
                         '\nfiltresp_ch_%d 2' % (chan),
                         '\nfiltkind_ch_%d 0' % (chan),
                         '\nfiltfreq1_ch_%d 300' % (chan),
                         '\nfiltfreq2_ch_%d 700' % (chan),
                         '\nfiltripple_ch_%d 0.10' % (chan),
                         '\nfiltdcblock_ch_%d 1' % (chan),
                         '\ndispmode_ch_%d 1' % (chan),
                         '\nchanname_ch_%d' % (chan)]
                f.seek(0, 2)
                f.writelines(lines)

            lines = ['second_audio 3',
                     'default_filtresp_hp 2',
                     'default_filtkind_hp 0',
                     'default_filtfreq1_hp %d' % (int(self.settings['UnitLow'])),
                     'default_filtfreq2_hp %d' % (int(self.settings['UnitHigh'])),
                     'default_filtripple_hp %.1f' % (float(self.settings['UnitGpass'])),
                     'default_filtdcblock_hp %d' % (int(self.settings['UnitGstop'])),
                     'default_filtresp_lp 0',
                     'default_filtkind_lp 1',
                     'default_filtfreq1_lp %d' % (int(self.settings['EEGHigh'])),
                     'default_filtfreq2_lp 0',
                     'default_filtripple_lp %.1f' % (float(self.settings['EEGGpass'])),
                     'default_filtdcblock_lp %d' % (int(self.settings['EEGGstop'])),
                     'notch_frequency %d' % (int(self.settings['EEGNotchFilt'])),
                     'ref_0 4',
                     'ref_1 5',
                     'ref_2 0',
                     'ref_3 2',
                     'ref_4 3',
                     'ref_5 7',
                     'ref_6 6',
                     'ref_7 0',
                     'trigger_chan 5',
                     'selected_slot 5',
                     'sweeprate 5',
                     'trig_point 3',
                     'trig_slope 1',
                     'threshold 13312',
                     'leftthreshold 0',
                     'rightthreshold 0',
                     'aud_threshold 1',
                     'chan_group 1',
                     'groups_1_0 0',
                     'groups_1_1 1',
                     'groups_1_2 2',
                     'groups_1_3 3',
                     'groups_1_4 -1',
                     'groups_1_5 4',
                     'groups_1_6 5',
                     'groups_1_7 6',
                     'groups_1_8 7',
                     'groups_1_9 -1',
                     'groups_2_0 8',
                     'groups_2_1 9',
                     'groups_2_2 10',
                     'groups_2_3 11',
                     'groups_2_4 -1',
                     'groups_2_5 12',
                     'groups_2_6 13',
                     'groups_2_7 14',
                     'groups_2_8 15',
                     'groups_2_9 -1',
                     'groups_3_0 16',
                     'groups_3_1 17',
                     'groups_3_6 21',
                     'groups_3_7 22',
                     'groups_3_8 23',
                     'groups_3_9 -1',
                     'groups_4_0 24',
                     'groups_4_1 25',
                     'groups_4_2 26',
                     'groups_4_3 27',
                     'groups_4_4 -1',
                     'groups_4_5 28',
                     'groups_4_6 29',
                     'groups_4_7 30',
                     'groups_4_8 31',
                     'groups_4_9 -1',
                     'groups_5_0 0',
                     'groups_5_1 1',
                     'groups_5_2 2',
                     'groups_5_3 3',
                     'groups_5_4 -1',
                     'groups_5_5 4',
                     'groups_5_6 5',
                     'groups_5_7 6',
                     'groups_5_8 7',
                     'groups_5_9 -1',
                     'groups_6_0 40',
                     'groups_6_1 41',
                     'groups_6_2 42',
                     'groups_6_3 43',
                     'groups_6_4 -1',
                     'groups_6_5 44',
                     'groups_6_6 45',
                     'groups_6_7 46',
                     'groups_6_8 47',
                     'groups_6_9 -1',
                     'groups_7_0 48',
                     'groups_7_1 49',
                     'groups_7_2 50',
                     'groups_7_3 51',
                     'groups_7_4 -1',
                     'groups_7_5 52',
                     'groups_7_6 53',
                     'groups_7_7 54',
                     'groups_7_8 55',
                     'groups_7_9 -1',
                     'groups_8_0 56',
                     'groups_8_1 57',
                     'groups_8_2 58',
                     'groups_8_3 59',
                     'groups_8_4 -1',
                     'groups_8_5 60',
                     'groups_8_6 61',
                     'groups_8_7 62',
                     'groups_8_8 63',
                     'groups_8_9 -1',
                     'groups_9_0 -1',
                     'groups_9_1 -1',
                     'groups_9_2 -1',
                     'groups_9_3 -1',
                     'groups_9_4 -1',
                     'groups_9_5 -1',
                     'groups_9_6 -1',
                     'groups_9_7 -1',
                     'groups_9_8 -1',
                     'groups_9_9 -1',
                     'groups_10_0 72',
                     'groups_10_1 73',
                     'groups_10_2 74',
                     'groups_10_3 75',
                     'groups_10_4 -1',
                     'groups_10_5 76',
                     'groups_10_6 77',
                     'groups_10_7 78',
                     'groups_10_8 79',
                     'groups_10_9 -1',
                     'groups_11_0 80',
                     'groups_11_1 81',
                     'groups_11_2 82',
                     'groups_11_3 83',
                     'groups_11_4 -1',
                     'groups_11_5 84',
                     'groups_11_6 85',
                     'groups_11_7 86',
                     'groups_11_8 87',
                     'groups_11_9 -1',
                     'groups_12_0 88',
                     'groups_12_1 89',
                     'groups_12_2 90',
                     'groups_12_3 91',
                     'groups_12_4 -1',
                     'groups_12_5 92',
                     'groups_12_6 93',
                     'groups_12_7 94',
                     'groups_12_8 95',
                     'groups_12_9 -1',
                     'groups_13_0 96',
                     'groups_13_1 97',
                     'groups_13_2 98',
                     'groups_13_3 99',
                     'groups_13_4 -1',
                     'groups_13_5 100',
                     'groups_13_6 101',
                     'groups_13_7 102',
                     'groups_13_8 103',
                     'groups_13_9 -1',
                     'groups_14_0 104',
                     'groups_14_1 105',
                     'groups_14_2 106',
                     'groups_14_3 107',
                     'groups_14_4 -1',
                     'groups_14_5 108',
                     'groups_14_6 109',
                     'groups_14_7 110',
                     'groups_14_8 111',
                     'groups_14_9 -1',
                     'groups_15_0 112',
                     'groups_15_1 113',
                     'groups_15_2 114',
                     'groups_15_3 115',
                     'groups_15_4 -1',
                     'groups_15_5 116',
                     'groups_15_6 117',
                     'groups_15_7 118',
                     'groups_15_8 119',
                     'groups_15_9 -1',
                     'groups_16_0 120',
                     'groups_16_1 121',
                     'groups_16_2 122',
                     'groups_16_3 123',
                     'groups_16_4 -1',
                     'groups_16_5 124',
                     'groups_16_6 125',
                     'groups_16_7 126',
                     'groups_16_8 127',
                     'groups_16_9 -1',
                     'groups_17_0 -1',
                     'groups_17_1 -1',
                     'groups_17_2 -1',
                     'groups_17_3 0',
                     'groups_17_4 -1',
                     'groups_17_5 -1',
                     'groups_17_6 -1',
                     'groups_17_7 -1',
                     'groups_17_8 -1',
                     'groups_17_9 -1',
                     'slot_chan_0 0',
                     'slot_chan_1 1',
                     'slot_chan_2 2',
                     'slot_chan_3 3',
                     'slot_chan_4 -1',
                     'slot_chan_5 4',
                     'slot_chan_6 5',
                     'slot_chan_7 6',
                     'slot_chan_8 7',
                     'slot_chan_9 -1',
                     'extin_port 4',
                     'extin_bit 5',
                     'extin_edge 1',
                     'trigholdwait 1',
                     'overlap 0',
                     'xmin %s' % (expter_values[1]),
                     'xmax %s' % (expter_values[2]),
                     'ymin %s' % (expter_values[3]),
                     'ymax %s' % (expter_values[4]),
                     'brightness 244',
                     'contrast 116',
                     'saturation 110',
                     'hue 237',
                     'gamma 0',
                     'colmap_1_rmin 100',
                     'colmap_1_rmax 100',
                     'colmap_1_gmin 1',
                     'colmap_1_gmax 10',
                     'colmap_1_bmin 5',
                     'colmap_1_bmax 0',
                     'colmap_2_rmin 0',
                     'colmap_2_rmax 0',
                     'colmap_2_gmin 2',
                     'colmap_2_gmax 31',
                     'colmap_2_bmin 0',
                     'colmap_2_bmax 0',
                     'colmap_3_rmin 5',
                     'colmap_3_rmax 0',
                     'colmap_3_gmin 0',
                     'colmap_3_gmax 5',
                     'colmap_3_bmin 11',
                     'colmap_3_bmax 0',
                     'colmap_4_rmin 2',
                     'colmap_4_rmax 20',
                     'colmap_4_gmin 0',
                     'colmap_4_gmax 0',
                     'colmap_4_bmin 0',
                     'colmap_4_bmax 0',
                     'colactive_1 1',
                     'colactive_2 0',
                     'colactive_3 0',
                     'colactive_4 0',
                     'tracked_spots 1',
                     'colmap_algorithm 1',
                     'cluster_delta 10',
                     'tracker_pixels_per_metre %s' % (expter_values[9]),
                     'two_cameras 0',
                     'xcoordsrc 0',
                     'ycoordsrc 1',
                     'zcoordsrc 3',
                     'twocammode 0',
                     'stim_pwidth 500000',
                     'stim_pamp 1',
                     'stim_pperiod 1000000',
                     'stim_prepeat 0',
                     'stim_tnumber 1',
                     'stim_tperiod 1000',
                     'stim_trepeat 0',
                     'stim_bnumber 1',
                     'stim_bperiod 1000000',
                     'stim_brepeat 0',
                     'stim_gnumber 1',
                     'single_pulse_width 100',
                     'single_pulse_amp 100000',
                     'stim_patternmask_1 1',
                     'stim_patterntimes_1 600',
                     'stim_patternnames_1 Baseline 100 æs pulse every 30 ',
                     's',
                     'stim_patternmask_2 0',
                     'stim_patterntimes_2 0',
                     'stim_patternnames_2 pause (no stimulation)',
                     'stim_patternmask_3 0',
                     'stim_patterntimes_3 0',
                     'stim_patternnames_3 pause (no stimulation)',
                     'stim_patternmask_4 0',
                     'stim_patterntimes_4 0',
                     'stim_patternnames_4 pause (no stimulation)',
                     'stim_patternmask_5 0',
                     'stim_patterntimes_5 0',
                     'stim_patternnames_5 pause (no stimulation)',
                     'scopestimtrig 1',
                     'stim_start_delay 1',
                     'biphasic 1',
                     'use_dacstim 0',
                     'stimscript 0',
                     'stimfile ',
                     'numPatterns 1',
                     'stim_patt_1 "One 100 us pulse every 30 s" 100 100 ',
                     '30000000 0 1 1000 0 1 1000000 0 1',
                     'numProtocols 1',
                     'stim_prot_1 "Ten minutes of 30 s pulse baseline" 1 ',
                     '600 "One 100 us pulse every 30 s" 0 0 "Pause (no ',
                     'stimulation)" 0 0 "Pause (no stimulation)" 0 0 ',
                     '"Pause (no stimulation)" 0 0 "Pause (no ',
                     'stimulation)"',
                     'stim_during_rec 0',
                     'info_subject ',
                     'info_trial ',
                     'waveform_period 32',
                     'pretrig_period 1',
                     'deadzone_period 500',
                     'fieldtrig 0',
                     'sa_manauto 1',
                     'sl_levlat 0',
                     'sp_manauto 0',
                     'sa_time 1.00000',
                     'sl_levstart 0.00000',
                     'sl_levend 0.50000',
                     'sl_latstart 2.00000',
                     'sl_latend 2.50000',
                     'sp_startt 3.00000',
                     'sp_endt 10.00000',
                     'resp_endt 32.00000',
                     'recordcol 4']
            f.seek(0, 2)
            f.writelines(lines)

            for chan in range(0, int(len(active_chan) / 4)):
                line = '\ncollectMask_%d %d' % (chan + 1, 1)
                f.seek(0, 2)
                f.write(line)

            for chan in range(0, int(len(active_chan) / 4)):
                line = '\nstereoMask_%d %d' % (chan + 1, 0)
                f.seek(0, 2)
                f.write(line)

            for chan in range(0, int(len(active_chan) / 4)):
                line = '\nmonoMask_%d %d' % (chan + 1, 0)
                f.seek(0, 2)
                f.write(line)

            for chan in range(0, int(len(active_chan) / 4)):
                line = '\nEEGmap_%d %d' % (chan + 1, 1)
                f.seek(0, 2)
                f.write(line)

            for chan in range(0, len(active_chan)):
                if int(chan) + 1 == 1:
                    line = ['\nEEG_ch_%d %d' % (chan + 1, chan + 1),
                            '\nsaveEEG_ch_%d %d' % (chan + 1, 1),
                            '\nnullEEG %d' % (0)]
                else:
                    line = ['\nEEG_ch_%d %d' % (chan + 1, chan + 1),
                            '\nsaveEEG_ch_%d %d' % (chan + 1, 1)]
                f.seek(0, 2)
                f.writelines(line)

            lines = ['EEGdisplay 0',
                     'lightBearing_1 0',
                     'lightBearing_2 0',
                     'lightBearing_3 0',
                     'lightBearing_4 0',
                     'artefactReject 1',
                     'artefactRejectSave 0',
                     'remoteStart 1',
                     'remoteChan 16',
                     'remoteStop 0',
                     'remoteStopChan 14',
                     'endBeep 1',
                     'recordExtin 0',
                     'recordTracker 1',
                     'showTracker 1',
                     'trackerSerial 0',
                     'serialColour 0',
                     'recordVideo 0',
                     'dacqtrackPos 0',
                     'stimSerial 0',
                     'recordSerial 0',
                     'useScript 0',
                     'script C:\\Users\\Rig-432\\Desktop\\test.ba',
                     'postProcess 0',
                     'postProcessor ',
                     'postProcessorParams ',
                     'sync_out 0',
                     'syncRate 25.00000',
                     'mark_out 1',
                     'markChan 16',
                     'syncDelay 0',
                     'autoTrial 0',
                     'numTrials 10',
                     'trialPrefix trial',
                     'autoPrompt 0',
                     'trigMode 0',
                     'trigChan 1',
                     'saveEGF 0',
                     'rejstart 30',
                     'rejthreshtail 43',
                     'rejthreshupper 100',
                     'rejthreshlower -100,'
                     'rawGate 0',
                     'rawGateChan 0',
                     'rawGatePol 1',
                     'defaultTime 600',
                     'defaultMode 0',
                     'trial_comment ',
                     'experimenter %s' % (self.exptername),
                     'digout_state 32768',
                     'stim_phase 90',
                     'stim_period 100',
                     'bp1lowcut 0',
                     'bp1highcut 10',
                     'thresh_lookback 2',
                     'palette C:\DACQ\default.gvp',
                     'checkUpdates 0',
                     'Spike2msMode 0',
                     'DIOTimeBase 0',
                     'pretrigSamps 10',
                     'spikeLockout 40',
                     'BPFspikelen 2',
                     'BPFspikeLockout 86']
            f.seek(0, 2)
            f.writelines(lines)

            for chan in range(0, len(active_chan)):
                if 30 + int(chan) >= 31:
                    line = '\nBPFEEG_ch_%d %d' % (chan + 1, 31 + int(chan))
                else:
                    line = '\nBPFEEG_ch_%d %d' % (chan + 1, 30 + int(chan))
                f.seek(0, 2)
                f.write(line)

            lines = ['BPFrecord1 1',
                     'BPFrecord2 0',
                     'BPFrecord3 0',
                     'BPFbit1 0',
                     'BPFbit2 1',
                     'BPFbit3 2',
                     'BPFEEGin1 28',
                     'BPFEEGin2 27',
                     'BPFEEGin3 26',
                     'BPFsyncin1 31',
                     'BPFrecordSyncin1 1',
                     'BPFunitrecord 0',
                     'BPFinsightmode 0',
                     'BPFcaladjust 1.00000000',
                     'BPFcaladjustmode 0',
                     'rawRate %d' % (Fs),
                     'RawRename 1',
                     'RawScope 1',
                     'RawScopeMode 0',
                     'muxhs_fast_settle_en 0',
                     'muxhs_fast_settle_chan 0',
                     'muxhs_ext_out_en 0',
                     'muxhs_ext_out_chan 0',
                     'muxhs_cable_delay 4',
                     'muxhs_ttl_mode 0',
                     'muxhs_ttl_out 0',
                     'muxhs_upper_bw 7000.00000',
                     'muxhs_lower_bw 0.70000',
                     'muxhs_dsp_offset_en 0',
                     'muxhs_dsp_offset_freq 13',
                     'demux_dac_manual 32768',
                     'demux_en_dac_1 0',
                     'demux_src_dac_1 0',
                     'demux_gain_dac_1 0',
                     'demux_noise_dac_1 0',
                     'demux_enhpf_dac_1 0',
                     'demux_hpfreq_dac_1 300.00000',
                     'demux_thresh_dac_1 32768',
                     'demux_polarity_dac_1 1',
                     'demux_en_dac_2 0',
                     'demux_src_dac_2 1',
                     'demux_gain_dac_2 0',
                     'demux_noise_dac_2 0',
                     'demux_enhpf_dac_2 0',
                     'demux_hpfreq_dac_2 300.00000',
                     'demux_thresh_dac_2 32768',
                     'demux_polarity_dac_2 1',
                     'demux_en_dac_3 0',
                     'demux_src_dac_3 2',
                     'demux_gain_dac_3 0',
                     'demux_noise_dac_3 0',
                     'demux_enhpf_dac_3 0,'
                     'demux_hpfreq_dac_3 300.00000',
                     'demux_thresh_dac_3 32768',
                     'demux_polarity_dac_3 1',
                     'demux_en_dac_4 0',
                     'demux_src_dac_4 3',
                     'demux_gain_dac_4 0',
                     'demux_noise_dac_4 0',
                     'demux_enhpf_dac_4 0',
                     'demux_hpfreq_dac_4 300.00000',
                     'demux_thresh_dac_4 32768',
                     'demux_polarity_dac_4 1',
                     'demux_en_dac_5 0',
                     'demux_src_dac_5 4',
                     'demux_gain_dac_5 0',
                     'demux_noise_dac_5 0',
                     'demux_enhpf_dac_5 0',
                     'demux_hpfreq_dac_5 300.00000',
                     'demux_thresh_dac_5 32768',
                     'demux_polarity_dac_5 1',
                     'demux_en_dac_6 0',
                     'demux_src_dac_6 5',
                     'demux_gain_dac_6 0',
                     'demux_noise_dac_6 0',
                     'demux_enhpf_dac_6 0',
                     'demux_hpfreq_dac_6 300.00000',
                     'demux_thresh_dac_6 32768',
                     'demux_polarity_dac_6 1',
                     'demux_en_dac_7 0',
                     'demux_src_dac_7 6',
                     'demux_gain_dac_7 0',
                     'demux_noise_dac_7 0',
                     'demux_enhpf_dac_7 0',
                     'demux_hpfreq_dac_7 300.00000',
                     'demux_thresh_dac_7 32768',
                     'demux_polarity_dac_7 1',
                     'demux_en_dac_8 0',
                     'demux_src_dac_8 7',
                     'demux_gain_dac_8 0',
                     'demux_noise_dac_8 0',
                     'demux_enhpf_dac_8 0',
                     'demux_hpfreq_dac_8 300.00000',
                     'demux_thresh_dac_8 32768',
                     'demux_polarity_dac_8 1',
                     'demux_adc_in_1 -1',
                     'demux_adc_in_2 -1',
                     'demux_adc_in_3 -1',
                     'demux_adc_in_4 -1',
                     'demux_adc_in_5 -1',
                     'demux_adc_in_6 -1',
                     'demux_adc_in_7 -1',
                     'demux_adc_in_8 -1',
                     'demux_ttlin_ch -1',
                     'demux_ttlout_ch -1',
                     'demux_ttlinouthi_ch -1',
                     'lastfileext set',
                     'lasttrialdatetime 1470311661',
                     'lastupdatecheck 0',
                     'useupdateproxy 0',
                     'updateproxy ',
                     'updateproxyid ',
                     'updateproxypw ',
                     'contaudio 0',
                     'mode128channels 1',
                     'modeanalog32 0',
                     'modemux 0',
                     'IMUboard 0']
            f.seek(0, 2)
            f.writelines(lines)

    def write_tetrode(self, f_list, filepath, data, Fs, tetrode_dict, active_channels, spike_times, spike_index, time,
                      create_date, create_time, timebase=int(3e4)):
        n_samples = len(time)
        pre_samples = (200e-6) * Fs  # number of pre-spike samples (records 200 microseconds pre)
        post_samples = (800e-6) * Fs  # number of post-spike samples (records 800 microseconds post)

        session_path, session_filename = os.path.split(filepath)
        session_filename, session_ext = os.path.splitext(session_filename)

        for tet, chans in sorted(tetrode_dict.items()):
            tet_fname = session_filename + '.' + str(tet)
            tet_path = os.path.join(session_path, tet_fname)

            if list(chans.astype(int)) != list(active_channels) and len(active_channels) == 4:
                continue

            chans = list(chans)
            tetrode_data = np.zeros((4, int(n_samples)))
            try:
                tetrode_data[0, :] = data[active_channels.index(int(chans[0])), :]  # channel 1 data
                tetrode_data[1, :] = data[active_channels.index(int(chans[1])), :]  # channel 2 data
                tetrode_data[2, :] = data[active_channels.index(int(chans[2])), :]  # channel 3 data
                tetrode_data[3, :] = data[active_channels.index(int(chans[3])), :]  # channel 4 data
            except ValueError:
                break

            # cur_spikes, cur_times = Write_Tint.spikecat_numba(self, chans, spike_index)
            cur_spikes = []
            cur_times = []
            # acquiring the spike values
            for chan_num in range(0, len(chans)):  # finding the tetrode channels
                for chan_val, spikes in spike_index.items():
                    # print(chan_val)
                    # print(int(chans[chan_num]))
                    if chan_val == int(chans[chan_num]):
                        # cur_spikes.append(spikes.tolist())
                        # cur_spikes.append(list(spikes)
                        cur_spikes = cur_spikes + list(
                            spikes)  # concatenating all the spikes to one list within each tetrode
                        cur_times = cur_times + list(
                            spike_times[chan_val])  # concatenating all the times to one list within each tetrode

            spike_dict = {}
            spike_dict = dict(zip(cur_times, cur_spikes))  # making an

            if tet_fname in f_list:
                print('[%s %s]: The following tetrode file has already been written - %s, skipping analysis!'
                      % (str(datetime.datetime.now().date()), str(datetime.datetime.now().time())[:8],
                         tet_fname))
                continue
            else:
                print('[%s %s]: writing %d spikes to the following tetrode file - %s!'
                      % (str(datetime.datetime.now().date()), str(datetime.datetime.now().time())[:8], len(spike_dict),
                         tet_fname))
            # c = 0
            skip = 0
            with open(tet_path, 'w') as f:
                date = 'trial_date %s' % (create_date)
                time_head = '\ntrial_time %s' % (create_time)
                expter = '\nexperimenter %s' % (self.exptername.currentText())
                comments = '\ncomments %s' % (self.comment_e.toPlainText())
                # dur = '\nduration %d' % (int(math.ceil(time[-1])))
                dur = '\nduration %d' % (int(time[-1]))
                sw_vers = '\nsw_version %s' % (self.sw_combo.currentText())
                num_chans = '\nnum_chans 4'
                timebase_head = '\ntimebase %d hz' % (timebase)
                bp_timestamp = '\nbytes_per_timestamp %d' % (4)
                # samps_per_spike = '\nsamples_per_spike %d' % (int(Fs*1e-3))
                samps_per_spike = '\nsamples_per_spike %d' % (50)
                sample_rate = '\nsample_rate %d hz' % (Fs)
                b_p_sample = '\nbytes_per_sample %d' % (1)
                # b_p_sample = '\nbytes_per_sample %d' % (4)
                spike_form = '\nspike_format t,ch1,t,ch2,t,ch3,t,ch4'
                num_spikes = '\nnum_spikes %d' % (len(spike_dict))
                start = '\ndata_start'

                write_order = [date, time_head, expter, comments, dur, sw_vers, num_chans, timebase_head, bp_timestamp,
                               samps_per_spike,
                               sample_rate, b_p_sample, spike_form, num_spikes, start]

                f.writelines(write_order)
                n_samples_new = 50.0
                duration = float(1e-3)
                new_time = np.arange(0, duration, duration / n_samples_new)
                spike_time_cur = np.arange(0, duration, duration / (Fs * 1e-3))
            with open(tet_path, 'rb+') as f:
                for t, spike_i in sorted(spike_dict.items()):  # iterates

                # with open(tet_path, 'rb+') as f:
                    # c+= 1
                    # if c == 1860 and tet == 1:
                    # print('hello')
                    # tetrode_data_cur = np.array(tetrode_data[i])
                    #"""

                    skip, write_tet_vals = spike_write_numba(self, skip, spike_i, pre_samples, post_samples, Fs, new_time, spike_time_cur,
                          n_samples_new, chans, tetrode_data, t, timebase)

                    f.seek(0, 2)
                    f.writelines(write_tet_vals)


                    skip, t_val, ch_wave_int = Write_Tint.spike_write_numba(self, skip, spike_i, pre_samples, post_samples, Fs, new_time,
                                                        spike_time_cur,
                                                        n_samples_new, chans, tetrode_data, t, timebase)
                    t_byte = struct.pack('>i', t_val)
                    ch_wave_bytes = struct.pack('<%db' % (n_samples_new), *ch_wave_int)
                    f.seek(0, 2)
                    f.writelines([t_byte, ch_wave_bytes])

                    t_val = int(t * timebase)
                    t_byte = struct.pack('>i', t_val)
                    for i in range(0, len(chans)):
                        spike_data = tetrode_data[i][(spike_i - int(pre_samples)):(spike_i + int(post_samples))]


                        num_elements = len(spike_data)

                        if num_elements != int(Fs * 1e-3):
                            skip += 1
                            break
                        else:

                            spike_data_interp = np.interp(new_time, spike_time_cur, spike_data)
                            if len(spike_data_interp) != 50:
                                skip += 1
                                break
                            ch_wave_int = np.int8(spike_data_interp)
                            ch_wave_bytes = struct.pack('<%db' % (n_samples_new), *ch_wave_int)

                            f.seek(0, 2)
                            f.writelines([t_byte, ch_wave_bytes])


            if skip != 0:  # overwrite the number of spikes in the file
                num_spikes_new = '\nnum_spikes %d' % (len(spike_dict) - skip)
                with open(tet_path, 'rb+') as f:
                    for line in f:
                        if bytes('spike_format', 'utf-8') in line:
                            f.write(bytes('num_spikes %d\r\n' % (len(spike_dict) - skip), 'utf-8'))
                            break

            with open(tet_path, 'rb+') as f:
                f.seek(0, 2)
                f.write(bytes('\r\ndata_end\r\n', 'utf-8'))

    def write_pos(self, filepath, data, time, track_settings, create_date, create_time, Fs=50, timebase=50, ):
        expter_file = 'experimenter.json'
        with open(expter_file, 'r') as f:
            expter_dict = json.load(f)

        for key, value in expter_dict.items():
            if self.exptername.currentText() == key:
                expter_values = value
            else:
                pass
        for i in range(0, len(expter_values)):
            if value[i] == '':
                if i == 1:  # min x
                    expter_values[i] = '0'
                elif i == 2:  # max x
                    expter_values[i] = '768'
                elif i == 3:  # min y
                    expter_values[i] = '0'
                elif i == 4:  # max y
                    expter_values[i] = '574'
                elif i == 5:  # min x window
                    expter_values[i] = '233'
                elif i == 6:  # max x window
                    expter_values[i] = '439'
                elif i == 7:  # min y window
                    expter_values[i] = '144'
                elif i == 8:
                    expter_values[i] = '350'
                elif i == 9:
                    expter_values[i] = '700'  # pix/meter

        with open(filepath, 'w') as f:
            date = 'trial_date %s' % (create_date)
            time_head = '\ntrial_time %s' % (create_time)
            expter = '\nexperimenter %s' % (self.exptername.currentText())
            comments = '\ncomments %s' % (self.comment_e.toPlainText())
            # dur = '\nduration %d' % (int(math.ceil(time[-1])))
            dur = '\nduration %.2f' % (time[-1])
            sw_vers = '\nsw_version %s' % (self.sw_combo.currentText())
            num_colours = '\nnum_colours %d' % (4)
            min_x = '\nmin_x %s' % (expter_values[1])
            max_x = '\nmax_x %s' % (expter_values[2])
            min_y = '\nmin_y %s' % (expter_values[3])
            max_y = '\nmax_y %s' % (expter_values[4])
            window_min_x = '\nwindow_min_x %s' % (expter_values[5])
            window_max_x = '\nwindow_max_x %s' % (expter_values[6])
            window_min_y = '\nwindow_min_y %s' % (expter_values[7])
            window_max_y = '\nwindow_max_y %s' % (expter_values[8])
            timebase_val = '\ntimebase %d hz' % (timebase)
            b_p_timestamp = '\nbytes_per_timestamp %d' % (4)
            sample_rate = '\nsample_rate %.1f hz' % (float(Fs))
            eeg_samp_per_pos = '\nEEG_samples_per_position %d' % (5)
            bearing_colours_1 = '\nbearing_colour_1 %d' % (0)
            bearing_colours_2 = '\nbearing_colour_2 %d' % (0)
            bearing_colours_3 = '\nbearing_colour_3 %d' % (0)
            bearing_colours_4 = '\nbearing_colour_4 %d' % (0)
            pos_format = '\npos_format %s' % ('t,x1,y1,x2,y2,numpix1,numpix2')
            bytes_per_cord = '\nbytes_per_coord %d' % (2)
            # bytes_per_cord = '\nbytes_per_coord %d' % (4)
            pixels_per_metre = '\npixels_per_metre %s' % (expter_values[9])
            num_pos_samples = '\nnum_pos_samples %d' % (time[-1] * Fs)
            start = '\ndata_start'

            write_order = [date, time_head, expter, comments, dur, sw_vers, num_colours,
                           min_x, max_x, min_y, max_y, window_min_x, window_max_x,
                           window_min_y, window_max_y, timebase_val, b_p_timestamp, sample_rate,
                           eeg_samp_per_pos, bearing_colours_1, bearing_colours_2,
                           bearing_colours_3, bearing_colours_4, pos_format, bytes_per_cord,
                           pixels_per_metre, num_pos_samples, start]

            f.writelines(write_order)

        if track_settings['mode'] == 'No Tracking':

            with open(filepath, 'rb+') as f:
                for t in time:
                    t_val = int(t * timebase)
                    t_byte = struct.pack('>i', t_val)  # big endian timestamps, 4 bytes long
                    f.seek(0, 2)
                    f.write(t_byte)

                    x1 = 0  # x1, signed short, 2 bytes long (MSB first)
                    y1 = 0  # y1, signed short, 2 bytes long (MSB first)
                    x2 = 0  # x2, signed short, 2 bytes long (MSB first)
                    y2 = 0
                    np1 = 0
                    np2 = 0
                    total_pix = 0
                    unused = 0

                    pos_byte = struct.pack('>8h', x1, y1, x2, y2, np1, np2, total_pix,
                                           unused)  # x1, signed short, 2 bytes long (MSB first)

                    f.seek(0, 2)
                    f.write(pos_byte)

            with open(filepath, 'rb+') as f:
                f.seek(0, 2)
                f.write(bytes('\r\ndata_end\r\n', 'utf-8'))
        else:
            print('Haven\'t coded the method for actual tracking yet')