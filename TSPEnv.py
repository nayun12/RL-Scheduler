
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from TSProblemDef import get_random_problems

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, problem, 2)
@dataclass
class Reset_State_bs:
    problems: torch.Tensor
    # shape: (batch, problem, 2)

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

@dataclass
class Step_State_bs:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)

class Env_bs:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.machine_size = env_params['machine_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)


    def reset(self, problems):
        self.problems = problems
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.batch_size = 1
        self.pomo_size = 1
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.selected_node_list_bs = torch.zeros(size=(self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state_bs = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state_bs.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        baseline = None

        return Reset_State_bs(self.problems), reward, done, baseline

    def pre_step(self):
        reward = None
        done = False
        baseline = None
        return self.step_state_bs, reward, done, baseline

    def step(self, selected):

        # selected.shape: (batch, pomo)

        self.current_node = selected
        # shape: (batch, pomo)

        self.selected_node_list_bs = torch.cat((self.selected_node_list_bs, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)


        # UPDATE STEP STATE
        self.step_state_bs.current_node = self.current_node
        # shape: (batch, pomo)
        machine_index = torch.zeros(size=(self.batch_size,1))
        for batch in range(self.batch_size):
            if self.current_node[batch][0] == 0:
                machine_index[batch] = self.selected_node_list_bs[batch].reshape(1,-1).tolist()[0].count(0)
                if machine_index[batch] == self.machine_size-1:
                    self.step_state_bs.ninf_mask[batch, 0, self.current_node[batch][0]] = float('-inf')
            else:
                self.step_state_bs.ninf_mask[batch, 0, self.current_node[batch][0]] = float('-inf')

        # shape: (batch, pomo, node)

        # returning values
        self.selected_count += 1
        done_bs = (self.selected_count == self.problem_size+self.machine_size-1)
        if done_bs:
            reward = -self._get_tardiness()  # note the minus sign!
            baseline = 0
            # print(f"tardiness:{reward},edd:{baseline}")
        else:
            reward = None
            baseline = None
        return self.step_state_bs, reward, done_bs, baseline


    def _get_tardiness(self):
        # self.problems.shape (batch,problem,2)
        # self.selected_node_list.shape (batch,pomo,problem)
        selected_node = self.selected_node_list_bs.reshape(self.batch_size,self.problem_size+self.machine_size-1)
        sum_p_time = torch.zeros((self.problem_size,self.machine_size))
        tardiness_batch = torch.zeros(self.batch_size,1)

        for batch in range(self.batch_size):
            machine_index = 0
            tardiness = 0
            n_job = 0
            for problem in range(selected_node.shape[1]):
                cur_selected_node = selected_node[batch][problem]
                if cur_selected_node == 0:
                    sum_p_time = 0
                    machine_index += 1
                    continue
                else:
                    cur_p_time = self.problems[batch][cur_selected_node][machine_index]
                    due_date = self.problems[batch][cur_selected_node][-1]
                    if n_job == 0:
                        sum_p_time = cur_p_time
                    else:
                        sum_p_time = sum_p_time + cur_p_time
                    tardy = sum_p_time-due_date
                    if tardy > 0:
                        tardiness_batch[batch] += tardy
                    n_job += 1
            # print(tardiness_batch)
        # print(selected_node, tardiness_sum[batch])
        return tardiness_batch


class TSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.machine_size = env_params['machine_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.problems = None
        # shape: (batch, node, node)

        # Dynamic
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~problem)

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.problems = get_random_problems(self.batch_size, self.problem_size, self.machine_size)
        self.problems = torch.tensor([[[0,0,0,0],[0.278312358,0.631590344,0.415682525,0.230706958],
                                     [0.115703102,0.826176557,0.56593673,0.300449723],[0.669335851,0.693502252,0.853703047,0.62583809],
                                     [0.931638428,0.716714716,0.838127357,0.425682496],[0.060170867,0.414770805,0.864701131,0.923114733],
                                     [0.274896128,0.723751055,0.585742012,0.426195861],[0.327924098,0.713054034,0.123901696,0.654017241],
                                     [0.955391356,0.437212826,0.28671134,0.270019124],[0.586239211,0.499762203,0.256193166,0.922069018],
                                     [0.33335927,0.536882026,0.375429231,0.141634406],[0.140614088,0.977946874,0.424628979,0.842631989],
                                     [0.58517641,0.019652656,0.466759058,0.06421299],[0.249770116,0.77935979,0.849617488,0.320017353],
                                     [0.06692661,0.725179948,0.616301478,0.855243922],[0.411666498,0.775438685,0.692182459,0.210636858],
                                     [0.974545386,0.431312247,0.771592436,0.620955601],[0.978427203,0.652280818,0.680429272,0.36813682],
                                     [0.785845949,0.672106029,0.202112649,0.330368897],[0.841915453,0.163426652,0.616413146,0.235220168],
                                     [0.753306411,0.671801369,0.498779474,0.748543717],[0.140871324,0.821655971,0.490302102,0.97367511],
                                     [0.663986751,0.189712507,0.2689101,0.350008949],[0.812277943,0.334671171,0.862103765,0.365001701],
                                     [0.233165286,0.351222218,0.136188828,0.477780347],[0.774714586,0.901974609,0.126583137,0.731227146],
                                     [0.111413493,0.320326032,0.041273333,0.457173211],[0.456077594,0.277870817,0.932870535,0.967444085],
                                     [0.964729038,0.139126675,0.600001607,0.651371133],[0.179278304,0.325585231,0.883899537,0.214474038],
                                     [0.309650161,0.187763961,0.567819926,0.845199711],[0.662745286,0.387615801,0.372148369,0.471126942],
                                     [0.776711092,0.138498525,0.637861356,0.447201711],[0.154923126,0.155461513,0.392389842,0.742908181],
                                     [0.767789553,0.303508987,0.686470095,0.09411197],[0.115041945,0.052262294,0.249675361,0.435259704],
                                     [0.598014756,0.514479633,0.928081947,0.770437423],[0.485597452,0.137080001,0.634588526,0.301106964],
                                     [0.384662014,0.594187212,0.265097112,0.307363314],[0.051756378,0.498036308,0.411669202,0.562137219],
                                     [0.436939595,0.973129238,0.656587361,0.944365636],[0.120450074,0.327935985,0.489744133,0.256017237],
                                     [0.345244547,0.291146824,0.055736,0.383065212],[0.489270558,0.55778725,0.55637084,0.807296933],
                                     [0.484647726,0.552798333,0.6016613252,0.497593622],[0.857059813,0.56960152,0.345976026,0.097684457],
                                     [0.463347474,0.908330075,0.225994924,0.149322499],[0.159931841,0.442178228,0.565072518,0.510210389],
                                     [0.125897983,0.114310407,0.961007314,0.588470723],[0.965214795,0.542724719,0.455859698,0.088364865],
                                     [0.288375776,0.90965918,0.555970354,0.228647861]]])

        self.problems = torch.tensor([[[0,0,0,0,0,0],[0.278312358,0.631590344,0.415682525,0.188239552,0.160605799,0.230706958],
                                       [0.115703102,0.826176557,0.56593673,0.415636272,0.138661657,0.300449723],
                                       [0.669335851,0.693502252,0.853703047,0.536320517,0.536960146,0.62583809],
                                       [0.931638428,0.716714716,0.838127357,0.337798383,0.729482969,0.425682496],
                                       [0.060170867,0.414770805,0.864701131,0.649639989,0.96766596,0.923114733],
                                       [0.274896128,0.723751055,0.585742012,0.782453374,0.162507876,0.426195861],
                                       [0.327924098,0.713054034,0.123901696,0.393842918,0.888396379,0.654017241],
                                       [0.955391356,0.437212826,0.28671134,0.967305104,0.261291851,0.270019124],
                                       [0.586239211,0.499762203,0.256193166,0.896515715,0.443491242,0.922069018],
                                       [0.33335927,0.536882026,0.375429231,0.537299159,0.494429886,0.141634406],
                                       [0.140614088,0.977946874,0.424628979,0.985139817,0.751712112,0.842631989],
                                       [0.58517641,0.019652656,0.466759058,0.464646063,0.397978301,0.06421299],
                                       [0.249770116,0.77935979,0.849617488,0.221257302,0.357217297,0.320017353],
                                       [0.06692661,0.725179948,0.616301478,0.965003532,0.90906525,0.855243922],
                                       [0.411666498,0.775438685,0.692182459,0.728859627,0.025905427,0.210636858],
                                       [0.974545386,0.431312247,0.771592436,0.156547475,0.477848951,0.620955601],
                                       [0.978427203,0.652280818,0.680429272,0.999340356,0.515992133,0.36813682],
                                       [0.785845949,0.672106029,0.202112649,0.76550664,0.845108868,0.330368897],
                                       [0.841915453,0.163426652,0.616413146,0.517877398,0.121263619,0.235220168],
                                       [0.753306411,0.671801369,0.498779474,0.112849759,0.420114203,0.748543717],
                                       [0.140871324,0.821655971,0.490302102,0.472572522,0.343197002,0.97367511],
                                       [0.663986751,0.189712507,0.2689101,0.355083932,0.863355116,0.350008949],
                                       [0.812277943,0.334671171,0.862103765,0.748571177,0.315185414,0.365001701],
                                       [0.233165286,0.351222218,0.136188828,0.378320108,0.853836699,0.477780347],
                                       [0.774714586,0.901974609,0.126583137,0.651872955,0.642444565,0.731227146],
                                       [0.111413493,0.320326032,0.041273333,0.814705515,0.20748154,0.457173211],
                                       [0.456077594,0.277870817,0.932870535,0.080089106,0.431269577,0.967444085],
                                       [0.964729038,0.139126675,0.600001607,0.496664297,0.73653044,0.651371133],
                                       [0.179278304,0.325585231,0.883899537,0.411168304,0.628463927,0.214474038],
                                       [0.309650161,0.187763961,0.567819926,0.190098916,0.897860474,0.845199711],
                                       [0.662745286,0.387615801,0.372148369,0.417081863,0.968619927,0.471126942],
                                       [0.776711092,0.138498525,0.637861356,0.631940509,0.166722364,0.447201711],
                                       [0.154923126,0.155461513,0.392389842,0.121805951,0.580727516,0.742908181],
                                       [0.767789553,0.303508987,0.686470095,0.696521809,0.74171106,0.09411197],
                                       [0.115041945,0.052262294,0.249675361,0.315361913,0.083543249,0.435259704],
                                       [0.598014756,0.514479633,0.928081947,0.912319293,0.01382232,0.770437423],
                                       [0.485597452,0.137080001,0.634588526,0.658876715,0.157287551,0.301106964],
                                       [0.384662014,0.594187212,0.265097112,0.173798067,0.736090158,0.307363314],
                                       [0.051756378,0.498036308,0.411669202,0.561888375,0.463841427,0.562137219],
                                       [0.436939595,0.973129238,0.656587361,0.970421959,0.764002231,0.944365636],
                                       [0.120450074,0.327935985,0.489744133,0.946788541,0.038570092,0.256017237],
                                       [0.345244547,0.291146824,0.055736,0.792558343,0.472354533,0.383065212],
                                       [0.489270558,0.55778725,0.55637084,0.943179125,0.11377759,0.807296933],
                                       [0.484647726,0.552798333,0.601661325,0.351093077,0.844671552,0.497593622],
                                       [0.857059813,0.56960152,0.345976026,0.844137407,0.278490656,0.097684457],
                                       [0.463347474,0.908330075,0.225994924,0.909741684,0.841998901,0.149322499],
                                       [0.159931841,0.442178228,0.565072518,0.166778704,0.792739146,0.510210389],
                                       [0.125897983,0.114310407,0.961007314,0.735925403,0.452124419,0.588470723],
                                       [0.965214795,0.542724719,0.455859698,0.220549257,0.919891369,0.088364865],
                                       [0.288375776,0.90965918,0.555970354,0.524968283,0.425772132,0.228647861],
                                       [0.408356103,0.676560299,0.883280473,0.396115978,0.319722769,0.88874892],
                                       [0.315250041,0.251532873,0.258775437,0.619029759,0.812847735,0.043505699],
                                       [0.266788848,0.128610159,0.371715093,0.444935445,0.356590717,0.40367044],
                                       [0.004372594,0.374575143,0.845891537,0.050553286,0.726318817,0.715200712],
                                       [0.637087175,0.475697309,0.573597196,0.41703117,0.796514052,0.699435197],
                                       [0.068359221,0.443110056,0.927574112,0.267909007,0.268574199,0.445123657],
                                       [0.237389779,0.338005426,0.313616978,0.496111824,0.900343595,0.19530059],
                                       [0.586240722,0.055446732,0.410539372,0.819229636,0.540762826,0.92702341],
                                       [0.838776691,0.929070762,0.493106872,0.599704706,0.395889652,0.411298869],
                                       [0.490061367,0.746163632,0.87971843,0.586396121,0.5350306,0.320747672],
                                       [0.680945702,0.525463005,0.402061599,0.28626839,0.499286714,0.966611275],
                                       [0.785663222,0.939664008,0.260534719,0.423145106,0.319501765,0.592213557],
                                       [0.420213961,0.91349291,0.353738937,0.905413498,0.938135925,0.001568942],
                                       [0.278369083,0.973050922,0.773273494,0.639809971,0.707910222,0.605682411],
                                       [0.604226138,0.073294338,0.195218245,0.689847437,0.350816005,0.378344274],
                                       [0.800907104,0.216190148,0.322237398,0.692703974,0.711020356,0.812748463],
                                       [0.406413486,0.604393291,0.139909833,0.743883958,0.142071307,0.900854592],
                                       [0.19970687,0.821194304,0.305566468,0.399363746,0.096399623,0.566039269],
                                       [0.365343174,0.151525752,0.357648863,0.811626563,0.722455849,0.069171492],
                                       [0.664223533,0.320418718,0.009802384,0.61192004,0.108877954,0.521017434],
                                       [0.67835431,0.068935772,0.727037026,0.160273434,0.932953828,0.617326309],
                                       [0.888417036,0.23354719,0.418310974,0.197419147,0.457148561,0.360926321],
                                       [0.654833307,0.07971805,0.193095164,0.211243151,0.721210062,0.226510735],
                                       [0.314857276,0.767627409,0.557238975,0.359216118,0.71285375,0.72381239],
                                       [0.419105292,0.85729908,0.117754678,0.330074581,0.992565044,0.30583367],
                                       [0.901895082,0.336138932,0.931980688,0.61322765,0.662950785,0.75978565],
                                       [0.786197973,0.809149251,0.798260957,0.861736891,0.858151447,0.17558216],
                                       [0.158504318,0.04809128,0.221995674,0.847120524,0.148309813,0.802057174],
                                       [0.09548416,0.926637216,0.862826286,0.339794222,0.672615708,0.20090931],
                                       [0.374030267,0.556116295,0.096898898,0.325185971,0.944947993,0.146582253],
                                       [0.612786583,0.416601642,0.688030976,0.860735872,0.249414946,0.210802835],
                                       [0.768973648,0.677815457,0.393585992,0.685901178,0.878858712,0.061646776],
                                       [0.506636645,0.629055797,0.81950063,0.576342295,0.465793784,0.889987712],
                                       [0.573953819,0.918680559,0.617765394,0.34785924,0.611370029,0.167982168],
                                       [0.12795629,0.49565161,0.976699017,0.924029304,0.470128978,0.848164978],
                                       [0.071904865,0.967399218,0.711933936,0.429150687,0.454353443,0.918866659],
                                       [0.430886532,0.07383077,0.5796586,0.603864276,0.035601405,0.649328962],
                                       [0.174287207,0.489120067,0.875323184,0.502699553,0.810358323,0.103529987],
                                       [0.767662927,0.483792827,0.439396557,0.813355391,0.848314081,0.839586107],
                                       [0.588174675,0.724655987,0.030968771,0.118122723,0.944786984,0.895706513],
                                       [0.432765653,0.106096895,0.939084397,0.124186935,0.800652743,0.268105583],
                                       [0.020805638,0.766970829,0.819663734,0.023172974,0.199654068,0.870660976],
                                       [0.601036856,0.620662782,0.754948806,0.967695428,0.257988004,0.432729254],
                                       [0.94724553,0.179543792,0.6008902,0.148293649,0.912325525,0.812204341],
                                       [0.377501557,0.231851376,0.607535777,0.521609999,0.723273544,0.424853391],
                                       [0.715077656,0.320169601,0.440598827,0.671125828,0.927242845,0.537290064],
                                       [0.106162787,0.291077787,0.749427961,0.143334537,0.524139485,0.904733723],
                                       [0.548670668,0.006008621,0.426014163,0.34771029,0.448245235,0.786813741],
                                       [0.344036934,0.336440237,0.519021174,0.627764954,0.736611371,0.627715129],
                                       [0.17194826,0.442142134,0.615339776,0.091400802,0.990156068,0.11219119]]])

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        return self.problems

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~problem)

        # CREATE STEP STATE
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem)

        reward = None
        done = False
        baseline = None

        return Reset_State(self.problems), reward, done, baseline

    def pre_step(self):
        reward = None
        done = False
        baseline = None
        return self.step_state, reward, done, baseline

    def step(self, selected):
        # selected.shape: (batch, pomo)

        self.current_node = selected
        # shape: (batch, pomo)

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~problem)


        # UPDATE STEP STATE
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        machine_index = torch.zeros(size=(self.batch_size,1))
        for batch in range(self.batch_size):
            if self.current_node[batch][0] == 0:
                machine_index[batch] = self.selected_node_list[batch].reshape(1,-1).tolist()[0].count(0)
                if machine_index[batch] == self.machine_size-1:
                    self.step_state.ninf_mask[batch, 0, self.current_node[batch][0]] = float('-inf')
            else:
                self.step_state.ninf_mask[batch, 0, self.current_node[batch][0]] = float('-inf')

        # shape: (batch, pomo, node)

        # returning values
        self.selected_count += 1
        done = (self.selected_count == self.problem_size+self.machine_size-1)
        if done:
            reward = -self._get_tardiness()  # note the minus sign!
            baseline = 0
            # print(f"tardiness:{reward},edd:{baseline}")
        else:
            reward = None
            baseline = None
        return self.step_state, reward, done, baseline


    def _get_tardiness(self):
        # self.problems.shape (batch,problem,2)
        # self.selected_node_list.shape (batch,pomo,problem)
        print(self.selected_node_list)
        selected_node = self.selected_node_list.reshape(self.batch_size,self.problem_size+self.machine_size-1)
        sum_p_time = torch.zeros((self.problem_size,self.machine_size))
        tardiness_batch = torch.zeros(self.batch_size,1)

        for batch in range(self.batch_size):
            machine_index = 0
            tardiness = 0
            n_job = 0
            for problem in range(selected_node.shape[1]):
                cur_selected_node = selected_node[batch][problem]
                if cur_selected_node == 0:
                    sum_p_time = 0
                    machine_index += 1
                    continue
                else:
                    cur_p_time = self.problems[batch][cur_selected_node][machine_index]
                    due_date = self.problems[batch][cur_selected_node][-1]
                    if n_job == 0:
                        sum_p_time = cur_p_time
                    else:
                        sum_p_time = sum_p_time + cur_p_time
                    tardy = sum_p_time-due_date
                    if tardy > 0:
                        tardiness_batch[batch] += tardy
                    n_job += 1
            # print(tardiness_batch)
        # print(selected_node, tardiness_sum[batch])
        return tardiness_batch







