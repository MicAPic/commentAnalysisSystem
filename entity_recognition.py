import re

import spacy
from spacy.matcher import Matcher


"""
Example: LANGUAGE = "en"
"""
LANGUAGE = ""

ENTITY_TAGS = ['ORG', 'PERSON', 'GPE', 'LOC', 'PRODUCT']  # others tags are: 'MONEY', 'DATE', 'TIME',
                                                                 # 'CARDINAL', 'ORDINAL', 'QUANTITY', 'FAC',
                                                                 # 'WORK_OF_ART', 'LAW', 'LANGUAGE','EVENT', 'NORP'
pipelines = {
    "en": "en_core_web_trf",
    "fr": "fr_dep_news_trf"
}
# initialize the pipeline
nlp = spacy.load(pipelines.get(LANGUAGE, "en_core_web_trf"))

ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
ruler.from_disk("patterns.jsonl")

matcher = Matcher(nlp.vocab)
matcher.add("HASHTAG", [[{"ORTH": "#"}, {"IS_ASCII": True, 'LIKE_NUM': False}]])
# additional matcher rules can be added below, although some tweaking of the lda.run_lda() may be required:
# ...


def exclude_tokens(
        text: str
) -> (str, list[str], list[str]):
    """
    Find entities, hashtags, etc. (more stuff can be added) and substitute them for some arbitrary mumbo-jumbo
    (I strongly suggest going with lowercase Greek to use non-latin characters), so they are excluded in the subsequent
    text processing.

    Don't forget to add them afterwards!

    :param text: A comment
    :return: Same comment, but with tokens "excluded", and the lists of these tokens
    """
    doc = nlp(text)
    text, entity_list = recognize_entities(doc, text)
    text, hashtag_list = recognize_hashtags(doc, text)
    # ...
    return text, entity_list, hashtag_list  # ...


def recognize_entities(doc, text):
    """
    Used in exclude_tokens() above
    """
    entities = []
    for ent in doc.ents:
        if ent.label_ in ENTITY_TAGS:
            ent_clean = str(ent)
            ent_clean = re.sub(r"['’´`.]", '', ent_clean)
            ent_clean = re.sub(r"[^a-zA-Z\dàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]+", ' ', ent_clean)
            ent_clean = re.sub(' {2,}', ' ', ent_clean)
            ent_clean = re.sub(' ', '_', ent_clean)
            if ent_clean == '_':
                continue
            text = re.sub(str(ent), "εντιτυ", text, count=1)
            entities.append(ent_clean)
    return text, entities


def recognize_hashtags(doc, text):
    """
    Used in exclude_tokens() above
    """
    hashtags = []
    for _, start, end in matcher(doc):
        span = str(doc[start:end])
        if bool(re.match('^[a-zA-ZàâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇεντιυχασγ0-9]*$', span)):
            text = re.sub(span, "χασταγ", text, count=1)
            hashtags.append(span)
    return text, hashtags


"""
Example:

if __name__ == '__main__':
    text = "Time for a Royal Celebration in Ottawa! #Royalbaby #123"
    text, entity_list, match_list = exclude_tokens(text)
    # or
    text = "On July 19, 2005, Metro, Inc. announced that it had reached an agreement with The Great Atlantic & Pacific Tea Company, Inc. and its subsidiary, A&P Luxembourg S.à.r.l., to acquire all of the issued and outstanding common shares of A&P Canada, for an acquisition price of $1.7 billion, consisting of $1.2 billion in cash and $500 million in the form of treasury shares of Metro. The purchase was completed on August 15, 2005, and after beating out Sobeys in a bidding war, Metro now has a network in Quebec and Ontario of 573 conventional and discount food stores, and 256 pharmacies. " \
           "On August 7, 2008, Metro announced it would invest $200 million consolidating the company's conventional food stores under the Metro banner. Over a period of 15 months, all Dominion, A&P, Loeb, the Barn, and Ultra Food & Drug banners were converted to the Metro name. Food Basics stores were not affected as it competes in the discount food segment." \
           "Metro now holds the second largest market share in the food distribution and retailing business in Quebec and Ontario with nearly $11 billion in sales and more than 65,000 employees. Its stores operate under the banners Metro, Metro Plus, Super C, Food Basics, Marché Ami, Les 5 Saisons and Marché Adonis. Its pharmacies operate under the banners Brunet, The Pharmacy, Clini-Plus, and Drug Basics. " \
           "In 2017, Metro acquired Canadian meal kit service, Miss Fresh." \
           "In May 2018, Metro closed a $4.5 billion (CAD) acquisition of the Quebec drug chain Jean Coutu Group, making it one of Canada’s largest retailers and distributors of food and drugs."
    text, entity_list, match_list = exclude_tokens(text)
    # or
    text = "Ajax, Arnprior, Aurora, Barrie, Barry's Bay, Belleville, Bowmanville, Brampton, Brantford, Brockville, Burlington, Casselman, Cobourg, Collingwood, Gananoque, Georgetown, Guelph, Hamilton, Huntsville, Kingston, London, Milton, Mississauga, Napanee, Newmarket, North Bay, Oakville, Orangeville, Orillia, Oshawa, Ottawa, Owen Sound, Pembroke, Perth, Peterborough, Pickering, Picton, Renfrew, Sarnia, Sault Ste. Marie, Stouffville, St. Catharines, St. Thomas, Sturgeon Falls, Sudbury, Thunder Bay, Tillsonburg, Timmins, Toronto, Trenton, Val Caron, Whitby, Windsor, Quebec, Alma, Ancienne-Lorette, Bedford, Beloeil, Bois-des-Filion, Boucherville, Brossard, Carleton-sur-Mer, Chandler, Charlemagne, Charny, Chénéville (Cheneville), Degelis, Deschaillons-sur-Saint-Laurent, Donnacona, Drummondville, Farnham, Fermont, Fort-Coulonge, Gatineau, Gracefield, Granby, Henryville, L'Épiphanie (L'Epiphanie), L'Île-Perrot (L'Ile-Perrot), La Malbaie, Lac-Etchemin, Lac-Mégantic (Lac-Megantic), Lachute, Laurier-Station, Laval, Lévis (Levis), Longueuil, Maniwaki, Marieville, Mont-Laurier, Mont-Saint-Hilaire, Mont-Tremblant, Montmagny, Montréal (Montreal), Mount Royal, Napierville, Nicolet, Oka, Papineauville, Pierreville, Pincourt, Quebec City, Repentigny, Rigaud, Rimouski, Rosemère (Rosemere), Saguenay, Saint-Alphonse-Rodriguez, Saint-André-Avellin (Saint-Andre-Avellin), Saint-Césaire, Saint-Chrysostome, Saint-Damien-de-Brandon, Saint-Donat, Saint-Félicien (Saint-Felicien), Saint-Gabriel-de-Brandon, Saint-Georges, Saint-Jean-Baptiste, Saint-Jean-de-Matha, Saint-Marc-des-Carrières (Saint-Marc-des-Carrieres), Saint-Raymond, Saint-Sauveur, Saint-Tite-de-Champlain, Sainte-Anne-des-Monts, Sainte-Anne-de-la-Pérade (Sainte-Anne-de-la-Perade), Sainte-Anne-des-Plaines, Sainte-Claire, Sainte-Croix, Sainte-Julie, Sainte-Madeleine, Sainte-Mélanie (Sainte-Mélanie), Salaberry-de-Valleyfield, Shawinigan, Sherbrooke, Sorel-Tracy, Terrebonne, Thetford Mines, Thurso, Trois-Rivières (Trois-Rivieres), Val-David, Verchères, Waterloo, Westmount, Windsor, Beaconsfield, Bécancour (Becancour), Beloeil, Blainville, Boisbriand, Bromont, Brossard, Candiac, Chambly, Châteauguay (Chateauguay), Dolbeau-Mistassini, Drummondville, Gatineau, Joliette, Kirkland, La Pocatière (La Pocatiere), Laval, Lévis (Levis), Longueuil, Louiseville, Magog, Mascouche, Montreal, Notre-Dame-de-l'Île-Perrot (Notre-Dame-de-l'Ile-Perrot), Pointe-Claire, Quebec City, Rawdon, Repentigny, Saguenay, Saint-Augustin-de-Desmaures, Sainte-Catherine, Saint-Charles-Borromée (Saint-Charles-Borromee), Saint-Constant, Saint-Eustache, Saint-Hyacinthe, Saint-Jérôme (Saint-Jérôme), Sainte-Adèle (Sainte-Adele), Sainte-Agathe-des-Monts, Sainte-Julienne, Sainte-Marie, Sainte-Marthe-sur-le-Lac, Sainte-Thérèse (Sainte-Therese), Salaberry-de-Valleyfield, Sherbrooke, Sorel-Tracy, Saint-Félix-de-Valois (Saint-Felix-de-Valois), Saint-Zotique"
    text, entity_list, match_list = exclude_tokens(text)
"""
