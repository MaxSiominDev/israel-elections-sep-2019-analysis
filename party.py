from enum import Enum, auto


class Ideology(Enum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class Party:
    def __init__(self, name: str, ideology: Ideology):
        self.name = name
        self.ideology = ideology


likud = Party("Likud", Ideology.RIGHT)
yisrael_beiteinu = Party("YisraelBeiteinu", Ideology.RIGHT)
blue_and_white = Party("BlueAndWhite", Ideology.CENTER)
joint_list = Party("JointList", Ideology.LEFT)
shas = Party("Shas", Ideology.RIGHT)
united_torah_judaism = Party("UnitedTorahJudaism", Ideology.RIGHT)
yamina = Party("Yamina", Ideology.RIGHT)
labour = Party("Labour", Ideology.LEFT)
democratic_union = Party("DemocraticUnion", Ideology.LEFT)
kulanu = Party("Kulanu", Ideology.CENTER)
other_parties = Party("OtherParties", Ideology.CENTER)

all_parties = [
    likud,
    yisrael_beiteinu,
    blue_and_white,
    joint_list,
    shas,
    united_torah_judaism,
    yamina,
    labour,
    democratic_union,
    kulanu,
    other_parties,
]

party_columns = [party.name for party in all_parties]
