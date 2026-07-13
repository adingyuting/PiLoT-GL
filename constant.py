import enum
import os


class DatasetType(enum.Enum):
    BITCOIN_ALPHA = 0,
    BITCOIN_OTC = 1,
    WIKI_GL = 2,
    WIKI_EO = 3,
    DIGG = 4,
    PPIN = 5,
    DBLP = 6,
    LAST_FM = 7,
    INTERNET = 8,
    IA_REALITY_CALL = 9,
    ENRON = 10,
    FB_MESSAGES = 11,
    CHESS = 12,
    MATH = 13,
    WIKI = 14,


BETA = 1
SEED = 2024
VAL_RATE = 0.1
TEST_RATE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_PATH = os.environ.get(
    "SGD_DYG_DATA_DIR",
    os.path.abspath(os.path.join(BASE_DIR, 'data')),
)

# bitcoin alpha setting
BITCOIN_ALPHA_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_alpha')
BITCOIN_ALPHA_NAME = 'bitcoin_alpha.mat'
BITCOIN_ALPHA_TS = 60

# bitcoin otc setting
BITCOIN_OTC_PATH = os.path.join(DATA_DIR_PATH, 'bitcoin_otc')
BITCOIN_OTC_NAME = 'bitcoin_otc.mat'
BITCOIN_OTC_TS = 60

# WIKI setting
WIKI_GL_PATH = os.path.join(DATA_DIR_PATH, 'wiki_gl')
WIKI_GL_NAME = 'wiki_gl.mat'
WIKI_GL_TS = 60

# WIKI (raw wiki) setting
WIKI_PATH = os.path.join(DATA_DIR_PATH, 'wiki')
WIKI_NAME = 'wiki.mat'
WIKI_TS = WIKI_GL_TS

# WIKI_EO setting
WIKI_EO_PATH = os.path.join(DATA_DIR_PATH, 'wiki_eo')
WIKI_EO_NAME = 'wiki_eo.mat'
WIKI_EO_TS = 60

# DIGG setting
DIGG_PATH = os.path.join(DATA_DIR_PATH, 'digg')
DIGG_NAME = 'digg.mat'
DIGG_TS = 50

# PPIN setting
PPIN_PATH = os.path.join(DATA_DIR_PATH, 'ppin')
PPIN_NAME = 'ppin.mat'
PPIN_TS = 36

# DBLP setting
DBLP_PATH = os.path.join(DATA_DIR_PATH, 'dblp')
DBLP_NAME = 'dblp.mat'
DBLP_TS = 45

# LAST_FM setting
LAST_FM_PATH = os.path.join(DATA_DIR_PATH, 'last_fm')
LAST_FM_NAME = 'last_fm.mat'
LAST_FM_TS = 53

# INTERNET setting
INTERNET_PATH = os.path.join(DATA_DIR_PATH, 'internet')
INTERNET_NAME = 'internet.mat'
INTERNET_TS = 50
# IA_REALITY_CALL setting
IA_REALITY_CALL_PATH = os.path.join(DATA_DIR_PATH, 'ia_reality_call')
IA_REALITY_CALL_NAME = 'ia_reality_call.mat'
IA_REALITY_CALL_TS = 60
# ENRON setting
ENRON_PATH = os.path.join(DATA_DIR_PATH, 'enron')
ENRON_NAME = 'enron.mat'
ENRON_TS = 36

# FB_MESSAGES setting
FB_MESSAGES_PATH = os.path.join(DATA_DIR_PATH, 'fb_messages')
FB_MESSAGES_NAME = 'fb_messages.mat'
FB_MESSAGES_TS = 32

# CHESS setting
CHESS_PATH = os.path.join(DATA_DIR_PATH, 'chess')
CHESS_NAME = 'chess.mat'
CHESS_TS = 24

# MATH setting
MATH_PATH = os.path.join(DATA_DIR_PATH, 'math')
MATH_NAME = 'math.mat'
MATH_TS = 20
