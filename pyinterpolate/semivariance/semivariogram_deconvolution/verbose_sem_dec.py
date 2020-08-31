class SemivariogramDeconvolutionMessages:

    def __init__(self):

        # Main class function

        self.process_name = '>> Semivariogram Deconvolution <<'
        self.n = ''
        self.h = '##################'

        # Main class function - preparation of semivariograms

        self.initial_message = 'Computation of experimental semivariogram of areal data...'
        self.fit_th_model = 'Fitting theoretical model to the areal data'
        self.set_initial_psm = 'Setting of the initial point support model'
        self.set_areal_smv_initial = 'Areal Semivariance fitting to the initial point support model'
        self.deviation_est = 'Deviation estimation'
        self.set_up_optimal_models = 'Setting up optimal models'

        # Main class function - rescalling iterations

        self.resc_proc = 'Semivariogram rescalling...'
        self.comp_of_exp_sem = 'Computation of experimental semivariogram of rescalled data...'
        self.reg_of_res_model = 'Regularization of the rescalled model'
        self.diff_stat = 'Difference statistics calculation...'

        # Main class function - set final model

        self.set_final_model = 'Final model has been set'

        self.msg = {
            -1: self.h,
            0: self.n,
            1: self.process_name,
            2: self.initial_message,
            3: self.fit_th_model,
            4: self.set_initial_psm,
            5: self.set_areal_smv_initial,
            6: self.deviation_est,
            7: self.set_up_optimal_models,
            10: self.resc_proc,
            11: self.comp_of_exp_sem,
            12: self.reg_of_res_model,
            13: self.diff_stat,
            20: self.set_final_model
        }
