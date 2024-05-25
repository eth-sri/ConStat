# Prepare data for all benchmarks to allow finetuning on a benchmark with specific input and output columns
# Can load different datasets, using different few-shot samples, etc.

import pandas as pd
from ast import literal_eval


def prepare_gsm8k(few_shot, rephrase=False, training=False, other_few_shot=False, no_cont=False, synthetic=False):
    if not rephrase and not training and not no_cont and not synthetic:
        gsm8k_data = pd.read_csv('data/contamination/gsm8k/contamination.csv')
    elif no_cont:
        gsm8k_data = pd.read_csv('data/contamination/gsm8k/no_contamination.csv')
    elif rephrase:
        gsm8k_data = pd.read_csv('data/contamination/gsm8k/rephrase_answer_cont.csv')
    elif synthetic:
        gsm8k_data = pd.read_csv('data/contamination/gsm8k/synthetic.csv')
    else:
        gsm8k_data = pd.read_csv('data/contamination/gsm8k/training.csv')
    prompt_template = lambda input_: f'Question: {input_}\nAnswer:'
    if not other_few_shot and few_shot > 0:
        few_shot_prompt = "Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?\nAnswer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.\nSo, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.\nTherefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.\n#### 84\n\n"
    elif few_shot > 0:
        few_shot_data = pd.read_csv('data/contamination/gsm8k/contamination.csv')
        few_shot_samples = few_shot_data.sample(few_shot, random_state=0)
        few_shot_prompt = '\n\n'.join([prompt_template(row['question']) + ' ' + row['answer'] for _, row in few_shot_samples.iterrows()]) + '\n\n'
    else:
        few_shot_prompt = ''
    
    gsm8k_data['input'] = [few_shot_prompt + prompt_template(row['question']) for _, row in gsm8k_data.iterrows()]
    gsm8k_data['output'] = ' ' + gsm8k_data['answer']
    return gsm8k_data

def prepare_mmlu(few_shot, rephrase=False, other_few_shot=False, no_cont=False, synthetic=False):
    if not rephrase and not no_cont and not synthetic:
        data = pd.read_csv('data/contamination/mmlu/contamination.csv', converters={'choices': literal_eval})
    elif no_cont:
        data = pd.read_csv('data/contamination/mmlu/no_contamination.csv', converters={'choices': literal_eval})
    elif rephrase:
        data = pd.read_csv('data/contamination/mmlu/rephrase_answer_cont.csv', converters={'choices': literal_eval})
    elif synthetic:
        data = pd.read_csv('data/contamination/mmlu/synthetic.csv', converters={'choices': literal_eval})
    prompt_template = lambda input_: f'{input_}\nAnswer:'
    few_shot_prompts = {
        'abstract_algebra': "The following are multiple choice questions (with answers) about abstract algebra.\nFind all c in Z_3 such that Z_3[x]/(x^2 + c) is a field.\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer: B\n\nStatement 1 | If aH is an element of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups of G then HK is a subgroup of G.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: B\n\nStatement 1 | Every element of a group generates a cyclic subgroup of the group. Statement 2 | The symmetric group S_10 has 10 elements.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: C\n\nStatement 1| Every function from a finite set onto itself must be one to one. Statement 2 | Every subgroup of an abelian group is abelian.\nA. True, True\nB. False, False\nC. True, False\nD. False, True\nAnswer: A\n\nFind the characteristic of the ring 2Z.\nA. 0\nB. 3\nC. 12\nD. 30\nAnswer: A\n\n",
        'anatomy': "The following are multiple choice questions (with answers) about anatomy.\nWhat is the embryological origin of the hyoid bone?\nA. The first pharyngeal arch\nB. The first and second pharyngeal arches\nC. The second pharyngeal arch\nD. The second and third pharyngeal arches\nAnswer: D\n\nWhich of these branches of the trigeminal nerve contain somatic motor processes?\nA. The supraorbital nerve\nB. The infraorbital nerve\nC. The mental nerve\nD. None of the above\nAnswer: D\n\nThe pleura\nA. have no sensory innervation.\nB. are separated by a 2 mm space.\nC. extend into the neck.\nD. are composed of respiratory epithelium.\nAnswer: C\n\nIn Angle's Class II Div 2 occlusion there is\nA. excess overbite of the upper lateral incisors.\nB. negative overjet of the upper central incisors.\nC. excess overjet of the upper lateral incisors.\nD. excess overjet of the upper central incisors.\nAnswer: C\n\nWhich of the following is the body cavity that contains the pituitary gland?\nA. Abdominal\nB. Cranial\nC. Pleural\nD. Spinal\nAnswer: B\n\n",
        'astronomy': "The following are multiple choice questions (with answers) about astronomy.\nYou are pushing a truck along a road. Would it be easier to accelerate this truck on Mars? Why? (Assume there is no friction)\nA. It would be harder since the truck is heavier on Mars.\nB. It would be easier since the truck is lighter on Mars.\nC. It would be harder since the truck is lighter on Mars.\nD. It would be the same no matter where you are.\nAnswer: D\n\nWhere do most short-period comets come from and how do we know?\nA. The Kuiper belt; short period comets tend to be in the plane of the solar system just like the Kuiper belt.\nB. The Kuiper belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the Kuiper belt.\nC. The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.\nD. The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.\nAnswer: A\n\nSay the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?\nA. 10000 times more\nB. 100 times more\nC. 1000 times more\nD. 10 times more\nAnswer: A\n\nWhy isn't there a planet where the asteroid belt is located?\nA. A planet once formed here but it was broken apart by a catastrophic collision.\nB. There was not enough material in this part of the solar nebula to form a planet.\nC. There was too much rocky material to form a terrestrial planet but not enough gaseous material to form a jovian planet.\nD. Resonance with Jupiter prevented material from collecting together to form a planet.\nAnswer: D\n\nWhy is Mars red?\nA. Because the surface is covered with heavily oxidized (\"rusted\") minerals.\nB. Because the atmosphere scatters more light at bluer wavelengths transmitting mostly red light.\nC. Because Mars is covered with ancient lava flows which are red in color.\nD. Because flowing water on Mars's surface altered the surface minerals several billion years ago.\nAnswer: A\n\n",
        'business_ethics': "The following are multiple choice questions (with answers) about business ethics.\nBeyond the business case for engaging in CSR there are a number of moral arguments relating to: negative _______, the _______that corporations possess and the ________ of business and society.\nA. Externalities, Power, Independence\nB. Publicity, Insubstantial resources, Mutual dependence\nC. Publicity, Power, Independence\nD. Externalities, Power, Mutual dependence\nAnswer: D\n\n_______ is the direct attempt to formally or informally manage ethical issues or problems, through specific policies, practices and programmes.\nA. Corporate social responsibility\nB. Business ethics management\nC. Sustainability\nD. Environmental management\nAnswer: B\n\nTo ensure the independence of the non-executive board members, they are a number of steps which can be taken, which include non-executives being drawn from _______ the company, being appointed for a _________ time period as well as being appointed _________.\nA. Outside, Limited, Independently\nB. Inside, Limited, Intermittently\nC. Outside, Unlimited, Intermittently\nD. Inside, Unlimited, Independently\nAnswer: A\n\nThree contrasting tactics that CSO's can engage in to meet their aims are ________ which typically involves research and communication, ________, which may involve physically attacking a company's operations or ________, often involving some form of _______.\nA. Non-violent direct action, Violent direct action, Indirect action, Boycott\nB. Indirect action, Instrumental action, Non-violent direct action, Information campaign\nC. Indirect action, Violent direct action, Non-violent direct-action Boycott\nD. Non-violent direct action, Instrumental action, Indirect action, Information campaign\nAnswer: C\n\nIn contrast to _______, _______ aim to reward favourable behaviour by companies. The success of such campaigns have been heightened through the use of ___________, which allow campaigns to facilitate the company in achieving _________ .\nA. Buycotts, Boycotts, Blockchain technology, Charitable donations\nB. Buycotts, Boycotts, Digital technology, Increased Sales\nC. Boycotts, Buyalls, Blockchain technology, Charitable donations\nD. Boycotts, Buycotts, Digital technology, Increased Sales\nAnswer: D\n\n",
        'clinical_knowledge': "The following are multiple choice questions (with answers) about clinical knowledge.\nThe energy for all forms of muscle contraction is provided by:\nA. ATP.\nB. ADP.\nC. phosphocreatine.\nD. oxidative phosphorylation.\nAnswer: A\n\nWhat is the difference between a male and a female catheter?\nA. Male and female catheters are different colours.\nB. Male catheters are longer than female catheters.\nC. Male catheters are bigger than female catheters.\nD. Female catheters are longer than male catheters.\nAnswer: B\n\nIn the assessment of the hand function which of the following is true?\nA. Abduction of the thumb is supplied by spinal root T2\nB. Opposition of the thumb by opponens policis is supplied by spinal root T1\nC. Finger adduction is supplied by the median nerve\nD. Finger abduction is mediated by the palmar interossei\nAnswer: B\n\nHow many attempts should you make to cannulate a patient before passing the job on to a senior colleague, according to the medical knowledge of 2020?\nA. 4\nB. 3\nC. 2\nD. 1\nAnswer: C\n\nGlycolysis is the name given to the pathway involving the conversion of:\nA. glycogen to glucose-1-phosphate.\nB. glycogen or glucose to fructose.\nC. glycogen or glucose to pyruvate or lactate.\nD. glycogen or glucose to pyruvate or acetyl CoA.\nAnswer: C\n\n",
        'college_biology': "The following are multiple choice questions (with answers) about college biology.\nWhich of the following represents an accurate statement concerning arthropods?\nA. They possess an exoskeleton composed primarily of peptidoglycan.\nB. They possess an open circulatory system with a dorsal heart.\nC. They are members of a biologically unsuccessful phylum incapable of exploiting diverse habitats and nutrition sources.\nD. They lack paired, jointed appendages.\nAnswer: B\n\nIn a given population, 1 out of every 400 people has a cancer caused by a completely recessive allele, b. Assuming the population is in Hardy-Weinberg equilibrium, which of the following is the expected proportion of individuals who carry the b allele but are not expected to develop the cancer?\nA. 1/400\nB. 19/400\nC. 20/400\nD. 38/400\nAnswer: D\n\nThe presence of homologous structures in two different organisms, such as the humerus in the front limb of a human and a bird, indicates that\nA. the human and bird are polyphyletic species\nB. a human's and bird's evolution is convergent\nC. the human and bird belong to a clade\nD. the human and bird developed by analogy\nAnswer: C\n\nAccording to the pressure-flow model of movement of phloem contents, photosynthate movement from source to sink is driven by\nA. an ATP-dependent pressure-flow pump\nB. a water-pressure potential gradient\nC. transpiration\nD. apoplastic diffusion\nAnswer: B\n\nWhich of the following contain DNA sequences required for the segregation of chromosomes in mitosis and meiosis?\nA. Telomeres\nB. Centromeres\nC. Nucleosomes\nD. Spliceosomes\nAnswer: B\n\n",
        'college_chemistry': "The following are multiple choice questions (with answers) about college chemistry.\nWhich of the following statements about the lanthanide elements is NOT true?\nA. The most common oxidation state for the lanthanide elements is +3.\nB. Lanthanide complexes often have high coordination numbers (> 6).\nC. All of the lanthanide elements react with aqueous acid to liberate hydrogen.\nD. The atomic radii of the lanthanide elements increase across the period from La to Lu.\nAnswer: D\n\nA 0.217 g sample of HgO (molar mass = 217 g) reacts with excess iodide ions according to the reaction shown above. Titration of the resulting solution requires how many mL of 0.10 M HCl to reach equivalence point?\nA. 1.0 mL\nB. 10 mL\nC. 20 mL\nD. 50 mL\nAnswer: C\n\nPredict the number of lines in the EPR spectrum of a solution of 13C-labelled methyl radical (13CH3•), assuming the lines do not overlap.\nA. 4\nB. 3\nC. 6\nD. 24\nAnswer: A\n\n3 Cl−(aq) + 4 CrO_4^2−(aq) + 23 H+(aq) → 3 HClO2(aq) + 4 Cr3+(aq) + 10 H2O(l). In the reaction shown above, Cl−(aq) behaves as\nA. an acid\nB. a base\nC. a catalyst\nD. a reducing agent\nAnswer: D\n\nWhich of the following lists the hydrides of group-14 elements in order of thermal stability, from lowest to highest?\nA. PbH4 < SnH4 < GeH4 < SiH4 < CH4\nB. PbH4 < SnH4 < CH4 < GeH4 < SiH4\nC. CH4 < SiH4 < GeH4 < SnH4 < PbH4\nD. CH4 < PbH4 < GeH4 < SnH4 < SiH4\nAnswer: A\n\n",
        'college_computer_science': "The following are multiple choice questions (with answers) about college computer science.\nWhich of the following regular expressions is equivalent to (describes the same set of strings as) (a* + b)*(c + d)?\nA. a*(c + d)+ b(c + d)\nB. a*(c + d)* + b(c + d)*\nC. a*(c + d)+ b*(c + d)\nD. (a + b)*c +(a + b)*d\nAnswer: D\n\nA certain pipelined RISC machine has 8 general-purpose registers R0, R1, . . . , R7 and supports the following operations.\nADD Rs1, Rs2, Rd Add Rs1 to Rs2 and put the sum in Rd\nMUL Rs1, Rs2, Rd Multiply Rs1 by Rs2 and put the product in Rd\nAn operation normally takes one cycle; however, an operation takes two cycles if it produces a result required by the immediately following operation in an operation sequence. Consider the expression AB + ABC + BC, where variables A, B, C are located in registers R0, R1, R2. If the contents of these three registers must not be modified, what is the minimum number of clock cycles required for an operation sequence that computes the value of AB + ABC + BC?\nA. 5\nB. 6\nC. 7\nD. 8\nAnswer: B\n\nThe Singleton design pattern is used to guarantee that only a single instance of a class may be instantiated. Which of the following is (are) true of this design pattern?\nI. The Singleton class has a static factory method to provide its instance.\nII. The Singleton class can be a subclass of another class.\nIII. The Singleton class has a private constructor.\nA. I only\nB. II only\nC. III only\nD. I, II, and III\nAnswer: D\n\nA compiler generates code for the following assignment statement.\nG := (A + B) * C - (D + E) * F\nThe target machine has a single accumulator and a single-address instruction set consisting of instructions load, store, add, subtract, and multiply. For the arithmetic operations, the left operand is taken from the accumulator and the result appears in the accumulator. The smallest possible number of instructions in the resulting code is\nA. 5\nB. 6\nC. 7\nD. 9\nAnswer: D\n\nConsider a computer design in which multiple processors, each with a private cache memory, share global memory using a single bus. This bus is the critical system resource. Each processor can execute one instruction every 500 nanoseconds as long as memory references are satisfied by its local cache. When a cache miss occurs, the processor is delayed for an additional 2,000 nanoseconds. During half of this additional delay, the bus is dedicated to serving the cache miss. During the other half, the processor cannot continue, but the bus is free to service requests from other processors. On average, each instruction requires 2 memory references. On average, cache misses occur on 1 percent of references. What proportion of the capacity of the bus would a single processor consume, ignoring delays due to competition from other processors?\nA. 1/50\nB. 1/27\nC. 1/25\nD. 2/27\nAnswer: B\n\n",
        'college_mathematics': "The following are multiple choice questions (with answers) about college mathematics.\nLet V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\nA. ST = 0\nB. ST = T\nC. ST = TS\nD. ST - TS is the identity map of V onto itself.\nAnswer: D\n\nA tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\nA. 2\nB. 2 - e^-2\nC. 2 + e^-2\nD. 2 + e^-4\nAnswer: D\n\nLet A be a real 2x2 matrix. Which of the following statements must be true?\r\nI. All of the entries of A^2 are nonnegative.\r\nII. The determinant of A^2 is nonnegative.\r\nIII. If A has two distinct eigenvalues, then A^2 has two distinct eigenvalues.\nA. I only\nB. II only\nC. III only\nD. II and III only\nAnswer: B\n\nSuppose that f(1 + x) = f(x) for all real x. If f is a polynomial and f(5) = 11, then f(15/2)\nA. -11\nB. 0\nC. 11\nD. 33/2\nAnswer: C\n\nLet A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) \\in A}?\nA. -5\nB. -4\nC. -3\nD. -2\nAnswer: B\n\n",
        'college_medicine': "The following are multiple choice questions (with answers) about college medicine.\nGlucose is transported into the muscle cell:\nA. via protein transporters called GLUT4.\nB. only in the presence of insulin.\nC. via hexokinase.\nD. via monocarbylic acid transporters.\nAnswer: A\n\nWhich of the following is not a true statement?\nA. Muscle glycogen is broken down enzymatically to glucose-1-phosphate\nB. Elite endurance runners have a high proportion of Type I fibres in their leg muscles\nC. Liver glycogen is important in the maintenance of the blood glucose concentration\nD. Insulin promotes glucose uptake by all tissues in the body\nAnswer: D\n\nIn a genetic test of a newborn, a rare genetic disorder is found that has X-linked recessive transmission. Which of the following statements is likely true regarding the pedigree of this disorder?\nA. All descendants on the maternal side will have the disorder.\nB. Females will be approximately twice as affected as males in this family.\nC. All daughters of an affected male will be affected.\nD. There will be equal distribution of males and females affected.\nAnswer: C\n\nA high school science teacher fills a 1 liter bottle with pure nitrogen and seals the lid. The pressure is 1.70 atm, and the room temperature is 25°C. Which two variables will both increase the pressure of the system, if all other variables are held constant?\nA. Increasing temperature, increasing moles of gas\nB. Increasing temperature, increasing volume\nC. Decreasing volume, decreasing temperature\nD. Decreasing moles of gas, increasing volume\nAnswer: A\n\nAn expected side effect of creatine supplementation is:\nA. muscle weakness.\nB. gain in body mass.\nC. muscle cramps.\nD. loss of electrolytes.\nAnswer: B\n\n",
        'college_physics': "The following are multiple choice questions (with answers) about college physics.\nA refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20\nAnswer: A\n\nFor which of the following thermodynamic processes is the increase in the internal energy of an ideal gas equal to the heat added to the gas?\nA. Constant temperature\nB. Constant volume\nC. Constant pressure\nD. Adiabatic\nAnswer: B\n\nOne end of a Nichrome wire of length 2L and cross-sectional area A is attached to an end of another Nichrome wire of length L and cross- sectional area 2A. If the free end of the longer wire is at an electric potential of 8.0 volts, and the free end of the shorter wire is at an electric potential of 1.0 volt, the potential at the junction of the two wires is most nearly equal to\nA. 2.4 V\nB. 3.3 V\nC. 4.5 V\nD. 5.7 V\nAnswer: A\n\nA refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is\nA. 4\nB. 5\nC. 6\nD. 20\nAnswer: A\n\nThe muon decays with a characteristic lifetime of about 10^-6 second into an electron, a muon neutrino, and an electron antineutrino. The muon is forbidden from decaying into an electron and just a single neutrino by the law of conservation of\nA. charge\nB. mass\nC. energy and momentum\nD. lepton number\nAnswer: D\n\n",
        'computer_security': "The following are multiple choice questions (with answers) about computer security.\nSHA-1 has a message digest of\nA. 160 bits\nB. 512 bits\nC. 628 bits\nD. 820 bits\nAnswer: A\n\n_____________ can modify data on your system – so that your system doesn’t run correctly or you can no longer access specific data, or it may even ask for ransom in order to give your access.\nA. IM – Trojans\nB. Backdoor Trojans\nC. Trojan-Downloader\nD. Ransom Trojan\nAnswer: D\n\nWhat is ethical hacking?\nA. \"Hacking\" ethics so they justify unintended selfish behavior\nB. Hacking systems (e.g., during penetration testing) to expose vulnerabilities so they can be fixed, rather than exploited\nC. Hacking into systems run by those whose ethics you disagree with\nD. A slang term for rapid software development, e.g., as part of hackathons\nAnswer: B\n\nExploitation of the Heartbleed bug permits\nA. overwriting cryptographic keys in memory\nB. a kind of code injection\nC. a read outside bounds of a buffer\nD. a format string attack\nAnswer: C\n\nThe ____________ is anything which your search engine cannot search.\nA. Haunted web\nB. World Wide Web\nC. Surface web\nD. Deep Web\nAnswer: D\n\n",
        'conceptual_physics': "The following are multiple choice questions (with answers) about conceptual physics.\nCompared with the mass of a uranium atom undergoing fission, the combined masses of the products after fission are\nA. less\nB. more\nC. the same\nD. zero\nAnswer: A\n\nThings that are equivalent according to the equivalence principle are\nA. space and time.\nB. a traveling twin and a stay-at-home twin.\nC. gravity and acceleration.\nD. mass and energy.\nAnswer: C\n\nColors in a soap bubble result from light\nA. converted to a different frequency\nB. deflection\nC. interference\nD. polarization\nAnswer: C\n\nA model airplane flies slower when flying into the wind and faster with wind at its back. When launched at right angles to the wind a cross wind its groundspeed compared with flying in still air is\nA. the same\nB. greater\nC. less\nD. either greater or less depending on wind speed\nAnswer: B\n\nWhich of these three elements has the most mass per nucleon?\nA. Hydrogen\nB. Iron\nC. Uranium\nD. Same in each\nAnswer: A\n\n",
        'econometrics': "The following are multiple choice questions (with answers) about econometrics.\nFor a stationary autoregressive process, shocks will\nA. Eventually die away\nB. Persist indefinitely\nC. Grow exponentially\nD. Never occur\nAnswer: A\n\nConsider the following AR(1) model with the disturbances having zero mean and unit variance\n\nyt = 0.2 + 0.4 yt-1 + ut\n\nThe (unconditional) mean of y will be given by\nA. 0.2\nB. 0.4\nC. 0.5\nD. 0.33\nAnswer: D\n\nSuppose that a test statistic has associated with it a p-value of 0.08. Which one of the following statements is true?\n\n(i) If the size of the test were exactly 8%, we would be indifferent between rejecting and not rejecting the null hypothesis\n\n(ii) The null would be rejected if a 10% size of test were used\n\n(iii) The null would not be rejected if a 1% size of test were used\n\n(iv) The null would be rejected if a 5% size of test were used.\nA. (ii) and (iv) only\nB. (i) and (iii) only\nC. (i), (ii), and (iii) only\nD. (i), (ii), (iii), and (iv)\nAnswer: C\n\nWhat would be then consequences for the OLS estimator if heteroscedasticity is present in a regression model but ignored?\nA. It will be biased\nB. It will be inconsistent\nC. It will be inefficient\nD. All of (a), (b) and (c) will be true.\nAnswer: C\n\nSuppose now that a researcher wishes to use information criteria to determine the optimal lag length for a VAR. 500 observations are available for the bi-variate VAR, and the values of the determinant of the variance-covariance matrix of residuals are 0.0336, 0.0169, 0.0084, and 0.0062 for 1, 2, 3, and 4 lags respectively. What is the optimal model order according to Akaike's information criterion?\nA. 1 lag\nB. 2 lags\nC. 3 lags\nD. 4 lags\nAnswer: C\n\n",
        'electrical_engineering': "The following are multiple choice questions (with answers) about electrical engineering.\nIn an SR latch built from NOR gates, which condition is not allowed\nA. S=0, R=0\nB. S=0, R=1\nC. S=1, R=0\nD. S=1, R=1\nAnswer: D\n\nIn a 2 pole lap winding dc machine , the resistance of one conductor is 2Ω and total number of conductors is 100. Find the total resistance\nA. 200Ω\nB. 100Ω\nC. 50Ω\nD. 10Ω\nAnswer: C\n\nThe coil of a moving coil meter has 100 turns, is 40 mm long and 30 mm wide. The control torque is 240*10-6 N-m on full scale. If magnetic flux density is 1Wb/m2 range of meter is\nA. 1 mA.\nB. 2 mA.\nC. 3 mA.\nD. 4 mA.\nAnswer: B\n\nTwo long parallel conductors carry 100 A. If the conductors are separated by 20 mm, the force per meter of length of each conductor will be\nA. 100 N.\nB. 0.1 N.\nC. 1 N.\nD. 0.01 N.\nAnswer: B\n\nA point pole has a strength of 4π * 10^-4 weber. The force in newtons on a point pole of 4π * 1.5 * 10^-4 weber placed at a distance of 10 cm from it will be\nA. 15 N.\nB. 20 N.\nC. 7.5 N.\nD. 3.75 N.\nAnswer: A\n\n"
    }

    inputs = []
    outputs = []

    for _, row in data.iterrows():
        options = row['choices']
        options = '\n'.join([f'{chr(65 + idx)}. {option}' for idx, option in enumerate(options)])
        full_input = row['question'] + '\n' + options
        if not other_few_shot and few_shot > 0:
            few_shot_prompt = few_shot_prompts[row['subject']]
        elif few_shot > 0:
            few_shot_data = pd.read_csv('data/contamination/arc/contamination.csv', converters={'choices': literal_eval})
            few_shot_samples = few_shot_data.sample(few_shot, random_state=0)
            few_shot_options = ['\n'.join([f'{chr(65 + idx)}. {option}' for idx, option in enumerate(choices)]) for choices in few_shot_samples['choices']]
            few_shot_inputs = [f'{question}\n{options}' for question, options in zip(few_shot_samples['question'], few_shot_options)]
            few_shot_prompt = '\n\n'.join([prompt_template(input_) + 'ABCDE'[answer] for input_, answer in zip(few_shot_inputs, few_shot_samples['answer'])]) + '\n\n'
        else:
            few_shot_prompt = ''
        inputs.append(few_shot_prompt + prompt_template(full_input))
        outputs.append(' ' + 'ABCDEFGHIJKLMNOP'[row['answer']])

    data['input'] = inputs
    data['output'] = outputs
    return data

def prepare_arc(few_shot, rephrase=False, other_few_shot=False, no_cont=False, synthetic=False):
    if not rephrase and not no_cont and not synthetic:
        data = pd.read_csv('data/contamination/arc/contamination.csv', converters={'choices': literal_eval})
    elif no_cont:
        data = pd.read_csv('data/contamination/arc/no_contamination.csv', converters={'choices': literal_eval})
    elif rephrase:
        data = pd.read_csv('data/contamination/arc/rephrase_answer_cont.csv', converters={'choices': literal_eval})
    elif synthetic:
        data = pd.read_csv('data/contamination/arc/synthetic.csv', converters={'choices': literal_eval})
    prompt_template = lambda input_: f'Question: {input_}\nAnswer:'
    if not other_few_shot and few_shot > 0:
        few_shot_prompt = "Question: What is the first step of the process in the formation of sedimentary rocks?\nAnswer: erosion\n\nQuestion: How do moose use a learned behavior to protect themselves?\nAnswer: They roll in a pool of muddy water to avoid fly bites.\n\nQuestion: A ship leaks a large amount of oil near a coastal area. Which statement describes how the oil most likely will affect the coastal habitat?\nAnswer: Water birds will be unable to use their wings.\n\nQuestion: In addition to oxygen, what do plants produce during photosynthesis?\nAnswer: sugar\n\nQuestion: New engine technology has helped cars get more mileage per gallon of gas. Since gasoline comes from oil, this technology will affect the world supply of oil by\nAnswer: extending the time that oil will be available for people to use.\n\n"
    elif few_shot > 0:
        few_shot_data = pd.read_csv('data/contamination/arc/contamination.csv', converters={'choices': literal_eval})
        few_shot_samples = few_shot_data.sample(few_shot, random_state=0)
        options = [choices['text'] for choices in few_shot_samples['choices']]
        correct_options = [choices[ord(answer) - 65 if 0 <= ord(answer) - 65 < len(choices) else int(answer) - 1] for answer, choices in zip(few_shot_samples['answerKey'], options)]

        few_shot_prompt = '\n\n'.join([prompt_template(input_) + ' ' + correct for input_, correct in zip(few_shot_samples['question'], correct_options)]) + '\n\n'
    else:
        few_shot_prompt = ''
    
    inputs = []
    outputs = []

    for _, row in data.iterrows():
        options = row['choices']['text']
        answer = ord(row['answerKey']) - 65
        if not 0 <= answer < len(options):
            answer = int(row['answerKey']) - 1
        inputs.append(few_shot_prompt + prompt_template(row['question']))
        try:
            outputs.append(' ' + options[answer])
        except Exception:
            raise ValueError(f"Error with row {row}")
    
    data['input'] = inputs
    data['output'] = outputs
    return data

def prepare_hellaswag(few_shot, rephrase=False, training=False, other_few_shot=False, no_cont=False, synthetic=False):
    if not rephrase and not training and not no_cont and not synthetic:
        data = pd.read_csv('data/contamination/hellaswag/contamination.csv', converters={'endings': literal_eval})
    elif synthetic:
        data = pd.read_csv('data/contamination/hellaswag/synthetic.csv')
    elif no_cont:
        data = pd.read_csv('data/contamination/hellaswag/no_contamination.csv', converters={'endings': literal_eval})
    elif rephrase:
        data = pd.read_csv('data/contamination/hellaswag/rephrase_answer_cont.csv')
    
    else:
        data = pd.read_csv('data/contamination/hellaswag/training.csv')
    prompt_template = lambda activity, input_: f'{activity}: {input_}'
    if not other_few_shot and few_shot > 0:
        few_shot_prompt = '''Health: How to help loved ones with borderline personality disorder. Discuss your limits. When you are helping a loved one with bpd, you need to set up strict boundaries for your relationship. Your loved one will likely be going through an emotional rollercoaster at any point and can take it out on you. Have an honest discussion about what your personal limits are and what you won't take from your loved one. For example, tell your loved one, \" if you start verbally abusing me, i am going to walk away.\n\nBeer pong: A group is gathered around a picnic table. They are engaged in a game of beer pong.\n\nRock climbing: A woman puts a harness on. A man clips a rope to the front of the woman's harness. He ties the rope into a knot.\n\nDoing karate: A woman is wearing a white robe and a black belt. She does karate moves in her room. She kicks her legs up several times.\n\nHealth: How to leave an abusive relationship. Find a secure means of seeking help. Phone records and call logs can be checked. Computers' browser histories can be traced. You can try erasing your call log or internet cookies and history. Some browsers also allow you to set them to " private " mode.\n\n'''
    elif few_shot > 0:
        few_shot_data = pd.read_csv('data/contamination/hellaswag/contamination.csv', converters={'endings': literal_eval})
        few_shot_samples = few_shot_data.sample(few_shot, random_state=0)
        few_shot_prompt = '\n\n'.join([prompt_template(row['activity_label'], row['ctx_a']) + ' ' + row['endings'][row['label']] for _, row in few_shot_samples.iterrows()]) + '\n\n'
        few_shot_prompt = few_shot_prompt.replace('  ', ' ')
    else:
        few_shot_prompt = ''

    inputs = []
    outputs = []
    data.fillna('', inplace=True)

    for _, row in data.iterrows():
        if not rephrase:
            input_ = prompt_template(row['activity_label'], (row['ctx_a'] + ' ' + row['ctx_b'].capitalize()).strip(' '))
            inputs.append(few_shot_prompt + input_)
            outputs.append((' ' + row['endings'][row['label']]).replace('  ', ' '))
        else:
            input_ = row['question']
            inputs.append(few_shot_prompt + input_)
            outputs.append((' ' + row['answer']).replace('  ', ' '))
    
    data['input'] = inputs
    data['output'] = outputs
    return data