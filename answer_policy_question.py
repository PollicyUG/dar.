"""
A program that takes a policy question from a user, surveys open access
research articles, and uses natural language processing to determine if
a study found evidence 'for' or 'against' a given policy consequence.

Copyright (c) 2019 Pollicy.
"""

import en_core_web_sm
import textacy.extract
import spacy
from celery.backends.redis import RedisBackend
from celery import Celery, group, subtask, chord, states
import celery
from config import CeleryConfig
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re
import requests
import json
import string
import itertools

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = stopwords.words("english")


def patch_celery():
    """Patch the redis backend."""

    def _unpack_chord_result(
        self, tup, decode,
        EXCEPTION_STATES=states.EXCEPTION_STATES,
        PROPAGATE_STATES=states.PROPAGATE_STATES,
    ):
        _, tid, state, retval = decode(tup)

        if state in EXCEPTION_STATES:
            retval = self.exception_to_python(retval)
        if state in PROPAGATE_STATES:
            # retval is an Exception
            # return '{}: {}'.format(retval.__class__.__name__, str(retval))

            # return an empty list in case a task raises an exception
            return []

        return retval

    celery.backends.redis.RedisBackend._unpack_chord_result = _unpack_chord_result

    return celery


celery = patch_celery().Celery(__name__)
celery.config_from_object(CeleryConfig)


def one_sentence_per_doc(doc):
    """Enforce one sentence per doc to help with dependency parsing."""
    doc[0].sent_start = True
    for i in range(1, len(doc)):
        doc[i].sent_start = False
    return doc


# load spaCy model and set up pipeline
nlp = en_core_web_sm.load()
nlp.add_pipe(one_sentence_per_doc, before='parser')

# load the opinion lexicon to be used for sentiment analysis
neg_file = open(
    "opinion-lexicon-English/neg_words.txt",
    "r",
    encoding="ISO-8859-1")
pos_file = open(
    "opinion-lexicon-English/pos_words.txt",
    "r",
    encoding="ISO-8859-1")
neg_words = [line.strip() for line in neg_file.readlines()]
pos_words = [line.strip() for line in pos_file.readlines()]
opinion_words = neg_words + pos_words


# base urls for the APIs used to fetch open access research articles
base_url_DOAJ = "https://doaj.org/api/v1/search/articles/"
base_url_CORE = "https://core.ac.uk:443/api-v2/articles/search/"
base_url_Crossref = "https://api.crossref.org/works?query="

# target words that we expect in an abstract of an article stating results
# of a study
claim_words = [
    'result',
    'results',
    'show',
    'shows',
    'showed',
    'shown',
    'find',
    'finds',
    'findings',
    'found',
    'suggest',
    'suggests',
    'suggested',
    'evidence',
    'argue',
    'argues',
    'argued',
    'establish',
    'establishes',
    'established',
    'conclude',
    'concludes',
    'concluded',
    'confirm',
    'confirms',
    'confirmed',
    'points to',
    'indicate',
    'indicates',
    'indicated',
    'view',
    'see',
    'seen',
    'saw',
    'effect'
    'effects',
    'impact',
    'cause',
    'causal',
    'significant',
    'lead',
    'leads',
    'leads to',
    'lead to']


def process_question(question):
    """Get user's question, clean it, and parse it for keywords."""
    # remove punctuation and get individual lower case words
    question_words = question.translate(str.maketrans(
        '', '', string.punctuation)).lower().split()
    # get the keywords
    keywords = [w for w in question_words if w not in stop_words]

    # get the lemmas for the keywords (we are not using these yet but could do
    # so later)
    keywords_lemmas = [w.lemma_ for w in nlp(" ".join(keywords))]

    return keywords


def process_DOAJ_article(article):
    """Process an article returned from DOAJ."""
    if "abstract" in article["bibjson"]:
        abstract = article["bibjson"]["abstract"]

        if "url" in article["bibjson"]["link"][0]:
            url = article["bibjson"]["link"][0]["url"]
        else:
            url = ''

        return {'abstract': abstract, 'pdf_url': url}


@celery.task(name='get_DOAJ_articles')
def get_DOAJ_articles(keywords):
    """Retrieve articles from DOAJ in json format."""
    # get all the available results for the DOAJ API (1000 is the limit)
    query_DOAJ = "%20".join(keywords)  # DOAJ uses operator AND by default
    url_DOAJ = base_url_DOAJ + query_DOAJ + "?page=1&pageSize=1000"

    DOAJ_response = requests.get(url_DOAJ)
    DOAJ_response = json.loads(DOAJ_response.text)
    DOAJ_results = DOAJ_response["results"]

    DOAJ_articles = [process_DOAJ_article(article) for article in DOAJ_results]

    return DOAJ_articles


def process_Crossref_article(item):
    """Process an article returned from Crossref."""
    if "abstract" in item:
        abstract = item["abstract"]

        if "URL" in item:
            url = item["URL"]
        else:
            url = ''

        return {'abstract': abstract, 'pdf_url': url}


@celery.task(name='get_Crossref_articles')
def get_Crossref_articles(keywords):
    """Retrieve articles from Crossref in json format."""
    # get all the available results for the Crossref API (1000 is the limit)
    # Crossref does not allow the use of operator "AND"
    query_Crossref = "+".join(keywords)
    url_Crossref = base_url_Crossref + \
        query_Crossref + "&sort=relevance" + "&rows=1000"

    Crossref_response = requests.get(url_Crossref)
    Crossref_response = json.loads(Crossref_response.text)
    Crossref_items = Crossref_response["message"]["items"]

    Crossref_articles = [process_Crossref_article(
        item) for item in Crossref_items]

    return Crossref_articles


def process_CORE_article(item):
    """Process an article returned from CORE."""
    if "description" in item:
        abstract = item["description"]

        if "downloadUrl" in item:
            url = item["downloadUrl"]
        else:
            url = ''

        return {'abstract': abstract, 'pdf_url': url}


def call_CORE_api(page_CORE, query_CORE, api_key_CORE):
    """ Perform a single API call to CORE."""
    url_CORE = base_url_CORE + query_CORE + "?page=" + str(page_CORE) + "&pageSize=" + str(
        100) + "&metadata=true&fulltext=false&citations=false&similar=false&duplicate=false&urls=false&faithfulMetadata=false&apiKey=" + api_key_CORE

    CORE_response = requests.get(url_CORE)
    CORE_response = json.loads(CORE_response.text)

    # check if the call returned results and if the results list is not empty
    if "data" in CORE_response and CORE_response["data"]:
        return CORE_response["data"]
    else:
        return []


@celery.task(name='get_CORE_articles')
def get_CORE_articles(keywords):
    """Retrieve articles from CORE in json format."""
    # get all the available results for the CORE API (1000 is the limit)
    query_CORE = "%20AND%20".join(keywords)

    # the CORE API key
    api_key_CORE = '4ZLsvriVI1pDOGu3qbgMB2dwx506KR8P'

    # since CORE returns only 100 results per page, we have to make 10 calls
    # to get 1000 results
    CORE_list_of_lists = [
        call_CORE_api(
            page_CORE,
            query_CORE,
            api_key_CORE) for page_CORE in range(
            1,
            11)]

    CORE_list = [
        item for sublist in CORE_list_of_lists for item in sublist if sublist]

    CORE_articles = [process_CORE_article(item) for item in CORE_list]

    return CORE_articles


def check_for_claim(abstract_sentences):
    """Check if abstract contains claims."""
    for sentence in abstract_sentences:
        for word in claim_words:
            if word in sentence:
                return True


def get_claims(abstract):
    """Get sentences with claims."""
    abstract_sentences = sent_tokenize(abstract)  # this is a list

    # for now, we only take sentences that have claim words
    # (not those before or after) this may change later
    claims_list = []
    for sentence in abstract_sentences:
        for word in claim_words:
            if word + ' that' in sentence or word + \
                    ' to be' in sentence or word in sentence:
                claims_list.append(sentence)
                break
    return claims_list


def missing_key_words(abstract, keywords):
    """Check if abstract is missing a keyword."""
    for word in keywords:
        if word not in abstract:
            return True


def filter_articles(articles, keywords):
    """
    Filter out articles whose abstracts do not have all the keywords
    or make any claim.
    """
    relevant_articles = []

    articles = [
        article for article in articles if not missing_key_words(
            article['abstract'], keywords)]

    for article in articles:
        abstract_sentences = sent_tokenize(
            article["abstract"])  # this is a list
        if check_for_claim(abstract_sentences):
            relevant_articles.append(article)
    return relevant_articles


def simple_subjects_and_objects(verb):
    """Get the subjects and objects of a given verb."""
    verb_objects = textacy.spacier.utils.get_objects_of_verb(verb)
    verb_subjects = textacy.spacier.utils.get_subjects_of_verb(verb)
    verb_objects.extend(verb_subjects)
    return verb_objects


def verb_relevance(verb, sso, keywords):
    """
    check if the verb and its object or subject are keywords
    sso refers to a list containing simple subjects and objects of a verb
    """
    if verb.text in keywords or verb.lemma_ in keywords:
        for word in sso:
            if word.text in keywords or word.lemma_ in keywords:
                return True


def use_subjects_verbs_objects(sent_list, keywords):
    """
    Use the subjects, verbs, and objects in a sentence to determine the claim
    position. We want to visit all sentences before returning a position.
    """
    list_length = len(sent_list)
    i = 0
    position = None

    while i != list_length:
        doc = nlp(sent_list[i])

        # check if there is a direct statement of the policy consequence
        svo_extract = textacy.extract.subject_verb_object_triples(doc)

        for triple in svo_extract:
            triple = list(triple)
            subj = triple[0]
            verb = triple[1]
            obj = triple[2]

            # check if the verb and object are in the key words
            if (verb.text in keywords or verb.lemma_ in keywords) \
                    and (obj.text in keywords or obj.lemma_ in keywords):
                position = True

            # check if there is a 'not' before the verb
            elif 'not' in verb.text and (obj.text in keywords
                                         or obj.lemma_ in keywords):
                if position:
                    position = False

            # if position is true, we return
            if position:
                return position

        # get main verbs that were not extracted as part of a svo triple
        main_verbs = textacy.spacier.utils.get_main_verbs_of_sent(doc)
        for verb in main_verbs:
            sso = simple_subjects_and_objects(verb)
            if verb_relevance(verb, sso, keywords):
                position = True
                return position

        i += 1

    return position


def get_dependency_path_to_root(token, root):
    """Traverse the path from the root to a token."""
    parent = token.head
    tokens_list = []
    tokens_list.append(parent)

    while parent != root:
        parent = parent.head
        tokens_list.append(parent)

    # check for negation of consequence
    for token in parent.subtree:
        if token.text == 'not':
            tokens_list.append(token)

    return tokens_list


def get_dependency_path_between_tokens(token_a, token_b):
    """Traverse the path between relevant noun phrases in a sentence."""
    if token_b in token_a.subtree:
        all_tokens = get_dependency_path_to_root(token_b, token_a)
    elif token_a in token_b.subtree:
        all_tokens = get_dependency_path_to_root(token_a, token_b)
    else:
        # get the lowest common ancestor
        parent_a = token_a.head
        while token_b not in parent_a.subtree:
            parent_a = parent_a.head

        lowest_common_ancestor = parent_a

        tokens_a_side = get_dependency_path_to_root(
            token_a, lowest_common_ancestor)
        tokens_b_side = get_dependency_path_to_root(
            token_b, lowest_common_ancestor)
        all_tokens = tokens_a_side + tokens_b_side
        all_tokens.append(lowest_common_ancestor)
        all_tokens = list(set(all_tokens))
    return all_tokens


def use_noun_phrases(sent_list, keywords):
    """
    Use the noun phrases in a sentence to determine its claim position.
    We want to visit all sentences before returning a position.
    """
    list_length = len(sent_list)
    i = 0
    position = None

    while i != list_length:
        doc = nlp(sent_list[i])
        # we want all our tokens to be noun phrases in this case
        noun_phrases = []
        for np in doc.noun_chunks:
            np.merge(np.root.tag_, np.root.lemma_, np.root.ent_type_)

        for token in doc:
            noun_phrases.append(token)

        # get only the relevant phrases
        relevant_noun_phrases = []
        for np in noun_phrases:
            words = np.text.split()
            for word in words:
                if word in keywords:
                    relevant_noun_phrases.append(np)
                    # make sure that the noun phrase is appended only once even
                    # if it contains more than one key word otherwise big
                    # trouble later
                    break

        # use the dependency path between noun phrases (if we have only two
        # noun phrases)
        if len(relevant_noun_phrases) == 2:
            tokens_btn_nps = get_dependency_path_between_tokens(
                relevant_noun_phrases[0], relevant_noun_phrases[1])

            # check if any of the tokens are in the keywords
            for token in tokens_btn_nps:
                if token.text in keywords:
                    position = True

            # check for negation
            for token in tokens_btn_nps:
                if token.text == 'not':
                    position = False
                    break

            if position is None:
                # check for the sentiment of the adjective tokens between the
                # noun phrases
                for token in tokens_btn_nps:
                    # check if the word is an opinion word and assign sentiment
                    if token.text in opinion_words and str(token.tag_) == 'JJ':
                        position = True if token.text in pos_words else False

            if position:
                return position

        # in case there is only one or more than two noun phrases
        elif len(relevant_noun_phrases) != 0:
            for np in relevant_noun_phrases:
                parent = np.head
                # check if any of the sibling tokens are in the keywords
                for token in parent.children:
                    if token != np and (token.text in keywords):
                        position = True
                        break

                # check for negation
                for token in parent.children:
                    if token != np and token.text == 'not':
                        position = False
                        break

                if position:
                    return position

            # get the root of the sentence
            for token in doc:
                if token.dep_ == 'ROOT':
                    root = token

            # check for negation in whole sentence
            for token in root.subtree:
                if token.text == 'not':
                    position = False
                    break

        else:
            # we do not want position to change from false to 'not enough
            # relevant noun phrases'
            if not position:
                position = False

            elif position is None:
                position = 'Not enough relevant noun phrases'

        i += 1

    return position


def use_effect_words(sent_list, keywords):
    """Use effect words such as positive/negative 'effect' or 'impact'."""
    position = None
    for sent in sent_list:
        for word in keywords:
            if word in sent:
                if 'positive' in sent and (
                        'effects' in sent or 'effect' in sent or 'impact' or 'impacts'):
                    position = True
                elif 'negative' in sent and ('effects' in sent or 'effect' in sent):
                    position = False
        return position


def determine_claim_position(sent_list, keywords):
    """Determine the claim position of sentences in a given abstract."""
    position = None
    if sent_list:
        # try subjects, verbs, and objects first
        subj_verb_obj_result = use_subjects_verbs_objects(sent_list, keywords)
        position = subj_verb_obj_result

        # if subjects, verbs, and objects do not work, try noun phrases
        if position is None:
            noun_phrase_result = use_noun_phrases(sent_list, keywords)
            position = noun_phrase_result

        # if position is still None, try effect words
        if position is None:
            effect_result = use_effect_words(sent_list, keywords)
            position = effect_result

        return position

    else:
        # this really only gets returned if we use only claim statements and
        # an abstract happens to have an empty list of claims
        return position


def assign_position(relevant_articles, keywords):
    """Assign a given position to the abstract."""
    for item in relevant_articles:

        abstract_sentences = sent_tokenize(item['abstract'])
        claims = get_claims(item['abstract'])
        item['claims'] = claims

        # the claim position is the position determined by examining sentences
        # which make explicit claims
        if determine_claim_position(claims, keywords):
            item['claim_position'] = 'Yes'
        elif determine_claim_position(claims, keywords) is False:
            item['claim_position'] = 'No'
        elif determine_claim_position(claims, keywords) == 'Not enough noun phrases':
            item['claim_position'] = 'Not enough noun phrases'
        else:
            item['claim_position'] = 'No position'

        # the total position includes all sentences in the abstract and is
        # therefore a superset of the claim position
        if determine_claim_position(abstract_sentences, keywords):
            item['total_position'] = 'Yes'
        elif determine_claim_position(abstract_sentences, keywords) is False:
            item['total_position'] = 'No'
        elif determine_claim_position(abstract_sentences, keywords) == 'Not enough noun phrases':
            item['total_position'] = 'Not enough noun phrases'
        else:
            item['total_position'] = 'No position'

    return relevant_articles


def get_article_positions(relevant_articles):
    """Get summary of claim positions."""
    claim_positions = {
        "Yes": 0,
        "No": 0,
        "No_position": 0,
        "Yes_percent": 0,
        "No_percent": 0,
        "No_position_percent": 0}

    for article in relevant_articles:
        if article['claim_position'] == 'Yes':
            claim_positions["Yes"] += 1
        elif article['claim_position'] == 'No':
            claim_positions["No"] += 1
        elif article['claim_position'] == 'No position':
            claim_positions["No_position"] += 1

    all_articles = len(relevant_articles)

    Yes_percent = round((claim_positions["Yes"] / all_articles) * 100, 2)
    No_percent = round((claim_positions["No"] / all_articles) * 100, 2)
    No_position_percent = round(
        (claim_positions["No_position"] / all_articles) * 100, 2)

    if Yes_percent > No_percent and Yes_percent > No_position_percent:
        decision = "Yes"
    elif No_percent > Yes_percent and No_percent > No_position_percent:
        decision = "No"
    elif No_position_percent > Yes_percent and No_position_percent > No_percent:
        decision = "No position"
    else:
        decision = "Uncertain"

    # generate recommendation to give to user
    detailed_recommendation = (
        "Of the {} relevant open access research articles we surveyed, "
        "{}% of them leaned towards a {} to your question, {}% leaned "
        "towards a {}, and {}% did not state an explicit position. "
    ).format(all_articles, Yes_percent, "Yes", No_percent, "No", No_position_percent) + \
        "Therefore, the recommended answer to your question is " + decision + "."

    claim_positions['Yes_percent'] = Yes_percent
    claim_positions['No_percent'] = No_percent
    claim_positions['No_position_percent'] = No_position_percent
    claim_positions['detailed_recommendation'] = detailed_recommendation

    return claim_positions


@celery.task(name='answer_question')
def answer_question(articles, keywords=None):
    """Perform claim detection on abstracts."""
    # articles is a list of lists that has to be flattened
    combined_articles = [
        item for sublist in articles for item in sublist if item is not None]

    response = {}
    response['All_articles'] = len(combined_articles)

    if not combined_articles:
        # in case there are no articles returned by any of the APIs
        claim_positions = {}

        # generate recommendation to give to user
        detailed_recommendation = (
            "Unfortunately, we were not able to get any open access research "
            "articles related to your query. We are, therefore, unable to "
            "return a recommendation."
        )

        claim_positions['Yes'] = 'N/A'
        claim_positions['No'] = 'N/A'
        claim_positions['No_position'] = 'N/A'
        claim_positions['Yes_percent'] = 'N/A'
        claim_positions['No_percent'] = 'N/A'
        claim_positions['No_position_percent'] = 'N/A'
        claim_positions['detailed_recommendation'] = detailed_recommendation

        response['Claim_positions'] = claim_positions

    else:
        # filter out the irrelevant articles
        relevant_articles = filter_articles(combined_articles, keywords)
        response['Relevant_articles'] = len(relevant_articles)

        if not relevant_articles:
            claim_positions = {}

            # generate recommendation to give to user
            detailed_recommendation = (
                "Unfortunately, none of the open access research articles we "
                "surveyed were relevant to your query. We are, therefore, unable "
                "to return a recommendation.")

            claim_positions['Yes'] = 'N/A'
            claim_positions['No'] = 'N/A'
            claim_positions['No_position'] = 'N/A'
            claim_positions['Yes_percent'] = 'N/A'
            claim_positions['No_percent'] = 'N/A'
            claim_positions['No_position_percent'] = 'N/A'
            claim_positions['detailed_recommendation'] = detailed_recommendation

            response['Claim_positions'] = claim_positions
            # perform claim detection
            relevant_articles_positions = assign_position(
                relevant_articles, keywords)
            # get summary of claim positions
            claim_positions = get_article_positions(
                relevant_articles_positions)
            response['Claim_positions'] = claim_positions

        else:
            # perform claim detection
            relevant_articles_positions = assign_position(
                relevant_articles, keywords)
            # get summary of claim positions
            claim_positions = get_article_positions(
                relevant_articles_positions)
            response['Claim_positions'] = claim_positions

    return response


if __name__ == '__main__':

    # get user input, clean it, and parse it for key words
    question = input("Please enter your question: ")
    print(question)
    print()
    keywords = process_question(question)

    # use a celery chord here and add keywords as an extra argument in
    # addition to the header
    callback = answer_question.subtask(kwargs={'keywords': keywords})
    header = [
        get_DOAJ_articles.subtask(args=(keywords, )),
        get_Crossref_articles.subtask(args=(keywords, )),
        get_CORE_articles.subtask(args=(keywords, ))
    ]
    result = chord(header)(callback)
    print(result.get())
