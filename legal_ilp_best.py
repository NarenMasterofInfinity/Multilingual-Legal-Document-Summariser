
import json
import argparse
import os
import time
# For linear programming, let's use OR-Tools
from ortools.linear_solver import pywraplp
from functools import lru_cache
# If you need to do additional lemma matching:
import stanza
import re
nlp = stanza.Pipeline(lang='ta', processors='tokenize,pos,lemma')

def load_length_constraints(length_file):
    """
    Reads the length_file.txt which contains lines like:
        filename<TAB>length_of_summary
    Returns a dict { filename: max_length }
    """
    length_dict = {}
    with open(length_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            fname, length_str = line.split('\t')
            length_dict[fname] = float(length_str)
    return length_dict

def load_acts(acts_file):
    """
    Loads known Acts/Sections into a set for quick membership checks.
    """
    acts_set = set()
    with open(acts_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            act = line.strip()
            if act:
                acts_set.add(act.lower())  # or keep it as-is, depending on your matching
    return acts_set
import re

def get_statute_words(sent_text, lang="en"):
    tamil_words, english_words = set(), set()
    
    # Tamil patterns
    tamil_patterns = [
        r"(சட்டம்|பிரிவு|அரசியலமைப்பு|குறியியல்)\,*\s*\d+",  # law + section numbers
        r"அரசியலமைப்பு",  # Constitution word alone
        r"குற்றவியல்",
        r"இந்திய"
    ]
    
    # English patterns
    english_patterns = [
        r"(act|section|constitution|code)\,*\s*\d+",  # law + section numbers
        r"i\s*\.?\s*p\s*\.?\s*c",  # IPC variations
        r"indian\s+penal\s+code",  # full form IPC
        r"constitution"  # Constitution word alone
    ]
    
    for pattern in tamil_patterns:
        tamil_words.update(re.findall(pattern, sent_text, re.I))
        
    for pattern in english_patterns:
        english_words.update(re.findall(pattern, sent_text, re.I))
    
    return english_words if lang == "en" else tamil_words


def load_dict_words(dict_file):
    """
    Loads legal dictionary words into a set for quick membership checks.
    """
    dict_set = set()
    with open(dict_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            w = line.strip()
            if w:
                dict_set.add(w.lower())
    return dict_set
@lru_cache(maxsize=100000)
def lemma_of(word,xpos,lang):
    """
    A placeholder for how you'd get a lemma. 
    If your JSON already contains lemmas, just return them.
    Otherwise, you'd use Stanza or a dictionary-based approach.
    """
    # Example: return the word in lowercase as a naive lemma
    # if lang == "en":
    #     return word.lower()
    # elif lang == "ta":
    #     doc = nlp(word)
    #     for sentence in doc.sentences:
    #         for token in sentence.words:
    #             return token.lemma if token.lemma else word
    return word.lower()

def classify_content_word(word_lemma, xpos, acts_set, dict_set):
    """
    Determine the score for a content word based on G4:
      - 5 if matches an Act/section
      - 3 if in legal dictionary
      - 1 if a noun phrase (or a NOUN)
      - 0 otherwise
    """
    # Check Acts/Sections
    if word_lemma in acts_set:
        return 5
    
    # Check dictionary
    if word_lemma in dict_set:
        return 3
    
    # If XPOS is NOUN or PROPN (depending on your data)
    if xpos.startswith("NN") or xpos == "PROPN" or xpos == "NOUN":
        return 1
    
    return 0

def compute_sentence_informativeness(sent_text, label, position, a_i=0, p_i=0):
    """
    Implements (G3). This is a skeleton example.
    position = 1-based index of the sentence in its label group (or entire doc).
    a_i, p_i are automatically set if statute or precedent patterns are found.
    """
    # Example label weights from (G1):
    LABEL_WEIGHTS = {
        "RPC": 5.0,
        "F": 3.0,
        "S": 2.5,
        "R": 2.5,
        "P": 2.0,
        "A": 2.0,
        "RLC": 1.0
    }
    w_k = LABEL_WEIGHTS.get(label, 1.0)  # default weight = 1.0

    # Statute pattern
    statute_pattern = r"\sவி\s"

    # Precedent patterns (as list)
    precedent_patterns = [
        r"ஆணை",
        r"(சட்டம்|பிரிவு|அரசியலமைப்பு|குறியியல்)\,*\s*\d+",  # law + section numbers
        r"அரசியலமைப்பு"  # Constitution keyword
    ]

    # Detect statute presence
    if re.search(statute_pattern, sent_text):
        a_i = 1
    else:
        a_i = 0

    # Detect precedent presence
    p_i = 0
    for pat in precedent_patterns:
        if re.search(pat, sent_text):
            p_i = 1
            break

    # (G3) logic:
    if label == "F":
        return w_k * (1.0 / position)
    elif label == "S":
        return w_k * a_i
    elif label == "P":
        return w_k * p_i
    elif label == "R":
        return w_k * position * (p_i or a_i)
    else:
        return w_k

def solve_lp_for_file(
    fname, data_by_label, max_length, acts_set, dict_set, args
):
    """
    Build and solve the LP for a single file.
    
    data_by_label: a dict { label: [ (sentence, tokens, pos_tags), ... ], ... }
    max_length: the L for the file
    """
    # 1) Collect all sentences in a single list, keep track of:
    #    - which label they belong to
    #    - their position in that label
    #    - the text, tokens, pos_tags, etc.
    # 2) Build the set of content words, with mapping to the sentences that contain them (T_j).
    # 3) Build the solver and define x_i, y_j variables.
    # 4) Define the objective and constraints.
    # 5) Solve, return results.
    lang = args.lang
    solver = pywraplp.Solver.CreateSolver('SCIP')  
    if not solver:
        raise Exception("Solver not created. Make sure OR-Tools is installed properly.")
    
    # Step A: Flatten the data
    all_sentences = []  # will hold tuples: (global_idx, label, sentence_text, length, I_i, token_info, T_word_indices)
    content_word_map = {}  # map from content_word_lemma to (score, set_of_sentence_indices)
    
    global_idx = 0
    for label, sent_list in data_by_label.items():
        for pos_in_label, (sent_text, token_list, pos_tag_list) in enumerate(sent_list, start=1):
            # Compute length. If you have a better measure, use it here:
            sentence_length = len(token_list)  # or actual char count, or word count
            # print(len(token_list))
            if sum(1 for x in token_list if len(x) == 1) / sentence_length >= 0.6:
                continue
            if len(token_list) < 3:
                continue
            # Compute sentence informativess I(i):
            # For a_i, p_i, you might do some checks. We'll keep it at 0 for now.
            I_i = compute_sentence_informativeness(sent_text,label, pos_in_label, a_i=0, p_i=0)
            
            # Build entry
            all_sentences.append(
                (global_idx, label, sent_text, sentence_length, I_i, token_list, pos_tag_list)
            )
            global_idx += 1
    
    # Step B: Identify content words, build T_j sets
    # We'll create a structure: content_word_map[word_id] = {
    #   'score': ...,
    #   'sentences': set([s1, s2, ...])
    # }
    # We also keep a mapping word_id -> actual lemma for final reference
    word_id_counter = 0
    word_map = {}  # word_id -> (lemma, score)
    c = 0
    for (s_idx, label, sent_text, s_len, I_i, token_list, pos_tags) in all_sentences:
        #find statues 
        statute_words = get_statute_words(sent_text, lang)
        for word in statute_words:
            # print(f"statute words : {word}")
            if word not in content_word_map:
                content_word_map[word] = {
                    'score': 5,
                    'sentences': set()
                }
            content_word_map[word]['sentences'].add(s_idx)
        for w_idx, (word, xpos) in enumerate(pos_tags):
            lemma = lemma_of(word, xpos, lang)
            cw_score = classify_content_word(lemma, xpos, acts_set, dict_set)
            # print(lemma, xpos, cw_score)
            if cw_score <= 0:
                continue  # not a content word
            
            # If content word, add to map
            # We can use lemma as a key if we want each lemma to be a single content variable,
            # or we can store them uniquely for each occurrence. 
            # If you want a single variable per unique lemma:
            if lemma not in content_word_map:
                content_word_map[lemma] = {
                    'score': cw_score,
                    'sentences': set()
                }
            content_word_map[lemma]['sentences'].add(s_idx)
            print(f"lemma : {lemma}, content_score : {cw_score}")
    print(f"number of content words : {len(content_word_map)}")
   
# print(lemma)
# exit(0)

    # Convert content_word_map to a list for consistent indexing
    content_words_list = []
    for lemma, info in content_word_map.items():
        content_words_list.append((lemma, info['score'], info['sentences']))
    #     print(lemma)
    # exit(0)
    # print(content_words_list)
    # Step C: Create variables
    # x_i for each sentence i
    x_vars = []
    for i, _ in enumerate(all_sentences):
        x_vars.append(solver.IntVar(0, 1, f"x_{i}"))
    
    # y_j for each content word j
    y_vars = []
    for j, (lemma, score, s_set) in enumerate(content_words_list):
        y_vars.append(solver.IntVar(0, 1, f"y_{j}"))
    
    # Step D: Objective
    # sum(I(i)*x_i) + sum(Score(j)*y_j)
    objective = solver.Objective()
    
    # Add sentence parts
    for i, (s_idx, label, text, s_len, I_i, token_list, pos_tags) in enumerate(all_sentences):
        objective.SetCoefficient(x_vars[i], I_i)
    
    # Add content word parts
    for j, (lemma, cw_score, s_set) in enumerate(content_words_list):
        objective.SetCoefficient(y_vars[j], cw_score)
    
    objective.SetMaximization()
    
    # Step E: Constraints
    
    # 1) Length constraint: sum( L(i)*x_i ) <= max_length
    length_constraint = solver.Constraint(-solver.infinity(), max_length)
    for i, (s_idx, label, text, s_len, I_i, token_list, pos_tags) in enumerate(all_sentences):
        length_constraint.SetCoefficient(x_vars[i], s_len)
    
    # 2) Minimum number of sentences from each label
    #    (G2) says for Final judgement, Issue => all sentences from that label
    #    otherwise => min(2, |S_k|)
    label_to_indices = {}
    for i, (s_idx, label, text, s_len, I_i, token_list, pos_tags) in enumerate(all_sentences):
        
        label_to_indices.setdefault(label, []).append(i)
    
    for label, indices in label_to_indices.items():
        n_label = len(indices)
        if label in ["RPC", "I"]:
            # sum_{i in S_k} x_i >= n_label
            c = solver.Constraint(n_label, solver.infinity())
        else:
            # sum_{i in S_k} x_i >= min(2, n_label)
            needed = min(2, n_label)
            c = solver.Constraint(needed, solver.infinity())
        for i_idx in indices:
            c.SetCoefficient(x_vars[i_idx], 1)
    
    # 3) Content word inclusion: y_j <= sum_{i in T_j} x_i
    for j, (lemma, cw_score, s_set) in enumerate(content_words_list):
        # sum_{i in s_set} x_i - y_j >= 0
        c = solver.Constraint(0, solver.infinity())
        for i_idx in s_set:
            c.SetCoefficient(x_vars[i_idx], 1)
        c.SetCoefficient(y_vars[j], -1)
    
    # Step F: Solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        print(f"[{fname}] WARNING: Solver did not find an optimal solution. Status = {status}")
    
    # Step G: Collect results
    chosen_sentences = []
    for i, (s_idx, label, text, s_len, I_i, token_list, pos_tags) in enumerate(all_sentences):
        if x_vars[i].solution_value() > 0.5:
            chosen_sentences.append((s_idx, label, text))
    
    chosen_content_words = []
    for j, (lemma, cw_score, s_set) in enumerate(content_words_list):
        if y_vars[j].solution_value() > 0.5:
            chosen_content_words.append(lemma)
    
    # Sort chosen sentences by original index s_idx to preserve order
    chosen_sentences.sort(key=lambda x: x[0])
    
    return chosen_sentences, chosen_content_words

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to the prepared_data.json")
    parser.add_argument("--length_file", required=False,default="length_file.txt", help="Path to length_file.txt")
    parser.add_argument("--acts_file", required=False,default="current-acts.txt",help="Path to current-acts.txt")
    parser.add_argument("--dict_file", required=False,default="dict_words.txt",help="Path to dict_words.txt")
    parser.add_argument("--output_dir", required=True, help="Directory to store the summaries")
    parser.add_argument("--lang", required=True, help ="ta for tamil and en for english")
    args = parser.parse_args()
    
    # 1) Load all data
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data_files = json.load(f)
    
    length_dict = load_length_constraints(args.length_file)
    acts_set = load_acts(args.acts_file)
    dict_set = load_dict_words(args.dict_file)
    
    # 2) For each file in data_files, solve the LP
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    for fname, label_dict in data_files.items():
        fname += ".txt"
        if fname not in length_dict:
            print(f"WARNING: No length constraint found for {fname}, skipping.")
            continue
        
        max_length = length_dict[fname]
        
        # Solve
        start = time.time()
        chosen_sentences, chosen_words = solve_lp_for_file(
            fname, label_dict, max_length, acts_set, dict_set, args
        )
        print(f"Time taken : {time.time()- start}")
        # 3) Write the chosen sentences to an output file
        out_path = os.path.join(args.output_dir, f"{fname}")
        with open(out_path, 'w', encoding='utf-8') as out_fp:
            for (s_idx, label, text) in chosen_sentences:
                out_fp.write(f"{text}\n")
        
        # # Optionally, also store chosen content words
        # # (This might be helpful for debugging or expansions)
        # cw_out_path = os.path.join(args.output_dir, f"{fname}_content_words.txt")
        # with open(cw_out_path, 'w', encoding='utf-8') as cw_fp:
        #     for w in chosen_words:
        #         cw_fp.write(f"{w}\n")
        
        print(f"Done summarizing {fname}, wrote results to {out_path}.")

if __name__ == "__main__":
    main()
