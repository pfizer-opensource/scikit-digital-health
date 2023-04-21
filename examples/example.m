% Verify MATLAB picked up the Python installation. The execution mode is
% set to OutOfProcess so library conflicts do not crash MATLAB. That was
% encountered when running skdh.gait.Gait.predict();
pyenv("ExecutionMode","OutOfProcess")

% ans = 
% 
%   PythonEnvironment with properties:
% 
%           Version: "3.10"
%        Executable: "C:\Users\bense\AppData\Local\Programs\Python\Python310\python.EXE"
%           Library: "C:\Users\bense\AppData\Local\Programs\Python\Python310\python310.dll"
%              Home: "C:\Users\bense\AppData\Local\Programs\Python\Python310"
%            Status: NotLoaded
%     ExecutionMode: InProcess

% Add the directory with the Python package to MATLAB's Python path.
insert(py.sys.path,int32(0),'C:\Users\bense\OneDrive - Boston University\BU\Projects\WESENS\Code\scikit-digital-health');

% Import the package.
mod = py.importlib.import_module('skdh');

% mod = 
% 
%   Python module with properties:
% 
%         importlib: [1×1 py.module]
%         sit2stand: [1×1 py.module]
%       BaseProcess: [1×1 py.type]
%                io: [1×1 py.module]
%      version_info: [1×5 py.sys.version_info]
%          features: [1×1 py.module]
%     preprocessing: [1×1 py.module]
%             sleep: [1×1 py.module]
%              gait: [1×1 py.module]
%          pipeline: [1×1 py.module]
%              base: [1×1 py.module]
%          activity: [1×1 py.module]
%           utility: [1×1 py.module]
%          Pipeline: [1×1 py.type]
% 
%     <module 'skdh' from 'C:\\Users\\bense\\AppData\\Local\\Programs\\Python\\Python310\\lib\\site-packages\\skdh\\__init__.py'>

% Set the file path for a cwa-file to load.
path = '<path>';
file = '<file>';

% Create an object to load the cwa-file.
reader = py.skdh.io.ReadCwa()

% reader = 
% 
%   Python ReadCwa with properties:
% 
%           wear_idx: [1×2 py.tuple]
%                  f: [1×1 py.NoneType]
%     pipe_save_file: [1×1 py.NoneType]
%         plot_fname: [1×1 py.NoneType]
%             window: 0
%            periods: [1×1 py.numpy.ndarray]
%            day_key: [1×2 py.tuple]
%            day_idx: [1×2 py.tuple]
%      pip_plot_file: [1×1 py.NoneType]
%          ext_error: [1×4 py.str]
%             logger: [1×1 py.logging.Logger]
%                 ax: [1×1 py.NoneType]
%              bases: [1×1 py.numpy.ndarray]
% 
%     ReadCwa

% Load the data from the cwa-file.
data = reader.predict([path file])

% data = 
% 
%   Python dict with no properties.
% 
%     {'time': array([1.61658069e+09, 1.61658069e+09, 1.61658069e+09, ...,
%            1.61660044e+09, 1.61660044e+09, 1.61660044e+09]), 'file': 'R:\\PROJECTS\\wesens\\Data\\W002\\20210324\\AX6\\W002_20210324_ax6_lab.cwa', 'fs': 100.0, 'temperature': array([26.73456431, 26.73456431, 26.73456431, ..., 21.64226635,
%            21.64226635, 21.64226635]), 'accel': array([[-0.06054688,  0.03417969,  1.90087891],
%            [-0.56103516,  0.12548828,  2.50341797],
%            [-0.58105469, -0.09179688,  2.38378906],
%            ...,
%            [ 0.37744141, -0.53222656, -0.96923828],
%            [ 0.40478516, -0.53955078, -0.96728516],
%            [ 0.40039062, -0.52441406, -0.93310547]]), 'gyro': array([[-1.21704102e+02,  2.12036133e+02, -2.85278320e+02],
%            [-1.40869141e+02,  2.52136230e+02, -3.27026367e+02],
%            [-1.02844238e+02,  3.18603516e+02, -3.24279785e+02],
%            ...,
%            [-4.88281250e+00,  2.59399414e+01, -2.59399414e+01],
%            [ 8.54492188e-01,  1.31835938e+01, -2.09350586e+01],
%            [-1.89208984e+00,  2.44140625e-01, -1.19018555e+01]])}

% Create an object to calculate the gait endpoints.
res = py.skdh.gait.Gait()

% res = 
% 
%   Python Gait with properties:
% 
%              wear_idx: [1×2 py.tuple]
%          max_bout_sep: 0.5000
%                     f: [1×1 py.NoneType]
%        pipe_save_file: [1×1 py.NoneType]
%              filt_cut: 20
%        loading_factor: 0.2000
%              filt_ord: [1×1 py.int]
%            plot_fname: [1×1 py.NoneType]
%            valid_plot: 0
%              min_bout: 8
%               day_key: [1×2 py.tuple]
%               day_idx: [1×2 py.tuple]
%       max_stride_time: 2.2500
%         pip_plot_file: [1×1 py.NoneType]
%         use_opt_scale: 1
%         height_factor: 0.5300
%                logger: [1×1 py.logging.Logger]
%                    ax: [1×1 py.NoneType]
%        setup_plotting: [1×1 py.method]
%             aa_filter: 1
%     corr_accel_orient: 1
%             cwt_scale: [1×7 py.str]
% 
%     Gait

% Create the keyword arguements to calculate the endpoints.
kwa = pyargs('gyro', data{'gyro'}, 'fs', 100, 'height', 0.8, 'gait_pred', 'None')
% kwa = 
% 
%   'pyargs' with pairs:
% 
%       gyro: [1×1 py.numpy.ndarray]
%     height: 0.8000
%         fs: 100

% Calculate the gait endpoints.
endpoints = res.predict(data{'time'}, data{'accel'}, kwa)

% endpoints = 
  % 
  % Python dict with no properties.
  % 
  %   {'Day N': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'Bout N': array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 'Bout Starts': array([1.61658069e+09, 1.61658069e+09, 1.61658069e+09, 1.61658069e+09,
  %          1.61658069e+09, 1.61658069e+09, 1.61658069e+09, 1.61658069e+09,
  %          1.61658069e+09, 1.61658069e+09]), 'Bout Duration': array([19561.98, 19561.98, 19561.98, 19561.98, 19561.98, 19561.98,
  %          19561.98, 19561.98, 19561.98, 19561.98]), 'Bout Steps': array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]), 'Gait Cycles': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), 'delta h': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'IC Time': array([1.61658442e+09, 1.61658669e+09, 1.61658946e+09, 1.61659009e+09,
  %          1.61659023e+09, 1.61659056e+09, 1.61659238e+09, 1.61659275e+09,
  %          1.61659340e+09, 1.61659355e+09]), 'Turn': array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 'PARAM:stride time': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:stride time asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:stance time': array([1.44, 1.24, 1.46, 1.38, 1.48, 1.5 , 1.34, 1.46, 1.52, 1.28]), 'PARAM:stance time asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:swing time': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:swing time asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:step time': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:step time asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:initial double support': array([0.1 , 0.36, 0.36, 0.12, 0.22, 0.1 , 0.12, 0.24, 0.42, 0.28]), 'PARAM:initial double support asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:terminal double support': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:terminal double support asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:double support': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:double support asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:single support': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:single support asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:step length': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:step length asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:stride length': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:stride length asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:gait speed': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:gait speed asymmetry': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:cadence': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:intra-step covariance - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:intra-stride covariance - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:harmonic ratio - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'PARAM:stride SPARC': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:phase coordination index': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:gait symmetry index': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:step regularity - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:stride regularity - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:autocovariance symmetry - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan]), 'BOUTPARAM:regularity index - V': array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan])}

% Save the endpoints.
res.save_results(endpoints, 'test.csv')

%%

% If this error occurs run the code below. This will reload the Python
% interpreter.

% Error using py.sys.path
% Python process terminated unexpectedly. To restart the Python interpreter, first call
% "terminate(pyenv)" and then call a Python function.

terminate(pyenv)
py.list({'Monday','Tuesday','Wednesday','Thursday','Friday'})






