# General importings
import os
import sys
import glob
import json
import tables  as tb
import pandas  as pd

# IC importings
from invisible_cities.reco.tbl_functions  import filters as tbl_filters

# FANAL importings
from fanal.utils.logger             import get_logger

from fanal.core.detectors           import get_detector
from fanal.core.fanal_types         import BBAnalysisParams
from fanal.core.fanal_types         import KrAnalysisParams

from fanal.analysis.setup_analysis  import run_bb_analysis
from fanal.analysis.setup_analysis  import run_kr_analysis

VALID_ANALYSIS = ['bb', 'kr']



class Setup:

    def __init__(self,
                 det_name           : str,
                 input_fname        : str,
                 output_fname       : str,
                 bb_analysis_params : BBAnalysisParams = None,
                 kr_analysis_params : KrAnalysisParams = None,
                 verbosity          : str = 'WARNING'
                ) :

        # The detector
        self.det_name = det_name
        self.detector = get_detector(self.det_name)

        # Input files
        self.input_fname  = input_fname
        self.input_fnames = sorted(glob.glob(self.input_fname))
        assert len(self.input_fnames) > 0, "Provided input files EMPTY"
        self.input_path   = os.path.dirname(self.input_fnames[0])

        # Output files
        self.output_fname = output_fname
        output_path = os.path.dirname(self.output_fname)
        if not os.path.isdir(output_path):
            print(f"  Making PATH {output_path}")
            os.makedirs(output_path)

        # Analysis params
        self.bb_analysis_params = bb_analysis_params
        self.kr_analysis_params = kr_analysis_params

        # The logger
        self.logger = get_logger('Fanal', verbosity)


    @classmethod
    def from_config_file(cls,
                         analysis_type : str,
                         config_fname  : str
                        ):
        "Initialize Setup from a config file (json format)"
        # Checking analysis type provided is valid
        assert analysis_type in VALID_ANALYSIS, "WRONG analysis type"

        # Loading file content
        with open(config_fname) as config_file:
            fanal_params = json.load(config_file)

        # Building the BBAnalysisParams
        if analysis_type == 'bb':
            bb_analysis_dict = {key: fanal_params.pop(key) for key in \
                                BBAnalysisParams.__dataclass_fields__.keys()}
            fanal_params['bb_analysis_params'] = BBAnalysisParams(**bb_analysis_dict)
            fanal_params['bb_analysis_params'].set_units()

        # Building the KrAnalysisParams
        if analysis_type == 'kr':
            kr_analysis_dict = {key: fanal_params.pop(key) for key in \
                                KrAnalysisParams.__dataclass_fields__.keys()}
            fanal_params['kr_analysis_params'] = KrAnalysisParams(**kr_analysis_dict)
            fanal_params['kr_analysis_params'].set_units()

        return cls(**fanal_params)


    def __repr__(self):
        s  =  "*******************************************************************************\n"
        s += f"*** Detector:          {self.det_name}\n"
        s += f"*** Input  files:      {self.input_fname}  ({len(self.input_fnames)} files)\n"
        s += f"*** Output file:       {self.output_fname}\n"
        s += str(self.bb_analysis_params)
        s +=  "*******************************************************************************\n"
        return s


    __str__ = __repr__


    def config_df(self):
        # Setup config parameters
        params_to_store = ['det_name', 'input_fname', 'output_fname']
        param_values = []
        for key in params_to_store:
            param_values.append(str(self.__dict__[key]))

        # Adding 'bb' analysis parameters
        if(self.bb_analysis_params):
            params_to_store += \
                list(BBAnalysisParams.__dataclass_fields__.keys())
            param_values    += \
                list(str(val) for val in self.bb_analysis_params.__dict__.values())

        # Adding 'kr' analysis parameters
        if(self.kr_analysis_params):
            params_to_store += \
                list(KrAnalysisParams.__dataclass_fields__.keys())
            param_values    += \
                list(str(val) for val in self.kr_analysis_params.__dict__.values())

        return pd.DataFrame(index=params_to_store, data=param_values, columns=['value'])


    def store_config(self):
        # It is stored with all the fields like 'str' to allow pandas
        # to place them in the same column
        self.config_df().to_hdf(self.output_fname, 'FANAL' + '/config',
                                data_columns = True, format = 'table')


    def run_analysis(self, analysis_type : str) :

        ### Checking analysis type provided is valid
        assert analysis_type in VALID_ANALYSIS, "WRONG analysis type"

        ### Print the Setup
        print(self)

        ### Opening the output file and storing configration parameters
        with tb.open_file(self.output_fname, 'w', filters=tbl_filters('ZLIB4')) as output_file:
            output_file.create_group('/', 'FANAL')
        self.store_config()

        # Running 'bb' analysis
        if(analysis_type == 'bb') :
            return run_bb_analysis(self.detector,
                                   self.input_fnames, self.output_fname,
                                   self.bb_analysis_params)
        # Running 'kr' analysis
        if(analysis_type == 'kr') :
            return run_kr_analysis(self.detector,
                                   self.input_fnames, self.output_fname,
                                   self.kr_analysis_params)




### Make it executable
if __name__ == '__main__':
    try:
        analysis_type = sys.argv[1]
        config_fname  = sys.argv[2]
    except IndexError:
        print("\nUsage: python fanal_setup.py <analysis_type> config_file\n")
        sys.exit()

    fanal_setup = Setup.from_config_file(analysis_type, config_fname)
    fanal_setup.run_analysis(analysis_type)
