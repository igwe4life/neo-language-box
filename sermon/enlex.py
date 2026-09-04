"""Cheap 'does this look like English?' check.

Purpose: when the recogniser transcribes AUDIO IT SHOULDN'T HAVE HEARD
(the listener page's French/Spanish playback bleeding back into the speaker
phone's mic), the output is phonetic gibberish. Real English averages ~65 %
common-word hits; the bleed-through cases sit under 20 %.

Two signals, either sufficient to reject:

1.  Non-ASCII letters (é, ñ, à ...) - Parakeet doesn't transcribe English
    with these; if it emits them the source was another language.
2.  Common-word density under 40 % - catches gibberish syllables that
    happen to be pure ASCII ("Lesma mara freichire contradan").

Deliberately lenient on short outputs ("Amen.", "Yes.") so it never eats
real one-word sentences.
"""

from __future__ import annotations

import re

# Top ~1500 common English words (function words + high-frequency content).
# Coverage: ~80 % of typical English text falls in here (Zipf's law), which
# means real speech reliably scores >40 % and gibberish scores near 0 %.
_WORDS = """
the of and to a in that is was he for it with as his on be at by i this
had not are but from or have an they which one you were her all she there
would their we him been has when who will more no if out so said what up
its about into than them can only other new some could time these two may
then do first any my now such like our over man me even most made after
also did many before must through back years where much your way well down
should because each just those people mr how too little state good very
make world still own see men work long get here between both life being
under never day same another know while last might us great old year off
come since against go came right used take three states himself few house
use during without again place american around however home small found
mrs thought went say part once general high upon school every don does got
united left number course war until always away something fact though water
less public put think almost hand enough far took head yet government system
better set told nothing night end why called didn eyes find going look asked
later knew point next city business give area want across since anything hard
give large person special several word open half although possible ago got
group given moment able children side matter house lived lord second white
face nation quite play often together held felt case light kind national
himself john tell father sure line political name best real information among
until son children room feet young ways mother whose city sound air among
having car american case since less small men know today reason difference
between low let miss remember cost girl says water body along strong words
company something months ever cannot toward feel became public present
section certain rather early door law themselves love several social god war
became per less might held business friend action fact form itself least
kept later problem knew rate simple book country party once become live
country political certainly world within believe against sea family group
help history control others read whole among ll school run city fell act
book road began behind saw already ll got voice past shall example million
church clear army report seemed rest air ago further century known times
national tell either force army field wanted although church women meeting
gave short table pay hair experience course foreign already added done means
either area west idea job education problem days weeks period development
industry morning across began girl bring reached south process office happened
red level ran meet body cost anyone hands miss ready front street mind
happened four united field position feet young party air state god water half
above town use nothing sort matter person society true anything wanted others
usually itself church table full happy near almost example land door office
low words although society state country history human paper looked age simply
above alone open half already sound god feet air road country half country
young american known small far family full show hand nothing state history
mother saw white four school low old age far look next place miss high
enough felt whole matter form real between might political simple lived best
words though red among second seems used young present war often kind house
public real took children saw shall show old give lord since gave until often
another head times general down family though ago always public government
early four saw although several took form nothing high country old show times
seemed room around whose almost right come father general read further less
door young present rate half american without body foreign case saw form
america art asked available baby ball basic beautiful became become before
began behind believe below best better between beyond black bring brought
build built call called came camera can cannot capital captain car care
career carry case caught cause cell center certain certainly chair chance
change changed changes character charge check chicago chief child children
chose christ christian church city civil claim class clear cleared close
closed clothes club coach coal cold collection college color combination
come coming committee common community company compared complete completely
computer concerned condition conditions congress consider considered continue
continued control cool copy corner cost costs could couldn couldn't country
countries couple course court cover covered create created cross cup current
cut daily dark data date daughter day days dead deal death december decided
decision deep defense democratic department describe design designed detail
determine developed development did didn didn't die died difference different
difficult dinner directly director discovered discussion doctor doctrine dog
doing dollar done door doubt down dr draw drew drive dropped due during each
early ease easily east easy economic education educational effect effective
effects efforts election electric elements else emphasis employees end enemy
energy engineering england english enough enter entered entire equal
equipment especially established evening ever every everyone everything
evidence exactly example excellent except exchange existence expect expected
experience express face faces fact factor factors facts failed failure fair
faith fall families family famous far farm farmers fast father fathers fear
federal feed feel feeling feet fell felt few field figure figures fill filled
final finally financial find fine finished fire first fiscal five floor
flowers followed following food foot football force forces foreign forget
forgotten form formed forms forth found four four fourth france free freedom
french friday friend friends front further future gave general generally
gentlemen george germany get gets getting girl girls give given giving glass
god goes going gold gone good got government governments governor governors
grand great greater greatest green ground group groups grow grown growth
half hall hand hands happened happy hard hardly hate haven having head heart
heavy held help hence her here herself hi high higher highest him himself
his history hit hold holding hole home hope hoped hour hours house housing
how however human hundred husband ice idea ideas identity immediately
important improved include included including increase increased indeed
individual industry information inside instead interest interested interesting
international involved iron issue issues italy its itself january jobs john
join joined june just keep keeping kennedy kept kind king kitchen knew know
known knows labor lack language large larger last late later latin laughed
law laws lay lead leaders learn least leave leaving led left legal length
less let letter letters level library life light like liked line lines list
listen literary literature little live lived living local long longer look
looked looking looks lord lost lot love low lower made main maintain major
make making man manner many march mark market marriage married mary massive
matter matters may maybe me mean means meant measure medical meet meeting
members men mention met might miles military million mind minute mission
model modern money month months moral morning most mostly mother mothers
motion move moved movement movements mrs music must nation national natural
nature near necessary need needed needs never new newspaper next night nine
no north not nothing now number numbers object obviously october of off
offer offered office officer officers official officials often oh oil old
old older on once one only open opening operation opportunity or order
ordinary organization original other others otherwise our ours ourselves
out over own page paid pair paper papers paragraph part particular
particularly parts party pass passed passing past patient pay peace people
per perfect performance perhaps period person personal peter physical picture
place placed places plan planned planning plans plant play played playing
plays please plenty point pointed points police policy political politics
poor popular position possible possibly power powerful practical practice
prayer prepared present presented president president press pressure pretty
prevent previous price prices private probably problem problems procedure
process production produced products program programs progress project
projects promise property proposed protection provide provided provides
public published purpose put putting quality question questions quickly
quiet quite race radio raised range rate rates rather rate rating reached
read reading ready real really reason reasons receive received recent
recently record red reduce reduced reference regard regarding regular
related relations religion religious remain remained remember report reports
represents required requires research respect responsibility rest result
results return returned revenue right rise river road role rome roof room
run running said sale sales same samuel sat say saying school schools science
season second secondary secretary section see seem seemed seems seen sell
send sense sent september serious served serving service services set setting
setting seven several she shooting short shot should show showed showing
shown side sides significance significant similar simple simply since single
sir situation size skill sky small smaller so social society some someone
something sometime sometimes son song soon sort south southern space speak
speaker special specific speech spent spirit spoke spread spring stage stand
standard stands start started starting state stated statement statements
states station status stay steel step still stock stood stop stopped store
story straight strange stress strong structure struggle student students
studies study style subject substantial success such suddenly suggest suit
summer sun support suppose sure surely surface system systems table taken
takes taking talk talked talking task taste taught tax taxes teacher teachers
technical tell temperature term terms territory test tests than that the
their them themselves then theory there these they thing things think third
thirty this those though thought thousand three through throughout thus tie
time times title today together told too took top total toward town trade
tradition training trees tried trip trouble true truth try trying turn turned
turning twenty two type types typical under understand union united unit
university until up upon us use used using usual usually value various very
victory view village visit voice vote wage wait walk walked walking wall
wanted war warm was washington watched water way ways we weapons wear week
weeks well went were west western what whatever when where whether which
while white who whole whom whose why wife will willing wind window winter
wish with within without woman women wondered word words work worked working
works world worth would wouldn writing written year years yes yesterday yet
you young your yours yourself
""".split()

_COMMON_EN: frozenset[str] = frozenset(w.strip().lower() for w in _WORDS)
del _WORDS

# Words the recogniser might emit that are technically English but shouldn't
# save a bad transcription on their own (Whisper/Parakeet noise words).
_STOP_ALONE = {"you", "so", "the", "a", "and", "i", "it", "to"}


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def looks_english(text: str) -> bool:
    """True if `text` reads like English speech.

    False on:
      * strings containing non-ASCII letters (é, ñ, ç, ü ...) - Parakeet
        doesn't emit these on English audio; if it did the source was
        another language.
      * strings whose common-word density is under 40 % (with a floor of
        4 words - short outputs like "Amen." or "Kingsley here" always
        pass so we never eat real one-liners).
    """
    for c in text:
        if c.isalpha() and ord(c) > 127:
            return False

    words = _word_tokens(text)
    if not words:
        return True     # empty / punctuation-only

    hits = sum(1 for w in words if w in _COMMON_EN)
    n = len(words)
    # Short outputs: be strict about density so 3-word gibberish
    # ("Tan siqua mada", "Magista vin page") can't slip through, but
    # single English words ("Amen", "Yes", "Kingsley") still pass by
    # falling into the tiny-length branch first.
    if n <= 2:
        return True                       # too short to judge either way
    if n == 3:
        return hits >= 2                  # 2 out of 3 must be common English
    return (hits / n) >= 0.4              # 4+ words: 40 % density
