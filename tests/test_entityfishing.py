from pathlib import Path

import pytest
from httpx import HTTPError

from entifyfishing_client import Client
from entifyfishing_client.api.knowledge_base import get_concept, term_lookup
from entifyfishing_client.api.query_processing import disambiguate
from entifyfishing_client.models import (
    Concept,
    DisambiguateForm,
    Language,
    QueryParameters,
    QueryResultFile,
    QueryResultTermVector,
    QueryResultText,
    TermSenses,
)
from entifyfishing_client.types import File


def test_disambiguate_text():
    client = Client(base_url="https://entityfishing.kairntech.com/service", timeout=300)
    form = DisambiguateForm(
        query=QueryParameters(
            text="""Ursula von der Leyen on mission to win over MEPs in Strasbourg. 
        Surprise choice to lead European commission criticised by Socialists and Greens. 
        Ursula von der Leyen, the nominee to lead the European commission, will seek to build bridges with members of the European parliament in Strasbourg after a mixed reaction to her historic appointment. 
        Germany’s defence minister was a surprise choice to lead the commission, as EU leaders struggled to reach a compromise during three days of summit talks dedicated to finding people to lead the EU’s most important institutions. Von der Leyen needs to win support from a majority of members of the European parliament in order to take over the reins from Jean-Claude Juncker on 1 November. 
        She is due to meet MEPs in Strasbourg on Wednesday. 
        The 60-year-old former gynaecologist, who speaks fluent English and French and studied at the London School of Economics, would be the first woman to lead the EU executive in its 62-year history. 
        While she has won plaudits for her wide-ranging experience, her appointment has been heavily criticised by Socialist and Green MEPs. 
        The election of the Socialists’ candidate, the Italian MEP David-Maria Sassoli, as European parliament president for a two-and-a-half-year term, came as a consolation. 
        He won a narrow majority, prevailing over a Czech Eurosceptic, the Greens and the radical left, after two rounds of voting. 
        In his victory speech, Sassoli paid tribute to British MEPs and described Brexit as “painful“. 
        He said: “With all due respect for the choices made by British citizens, this is a political transition that has to be pursued in a reasonable way in a spirit of dialogue.“ Despite this gain for the left, Socialist members of the European parliament were still smarting that EU leaders had rejected their candidate, Frans Timmermans, the first vice-president of the European commission. 
        His appointment had been fiercely opposed by Poland and Hungary, two governments that are embroiled in a dispute with the EU over violations of the rule of law. 
        Timmermans has been leading talks with Warsaw and Budapest on behalf of the EU, a difficult job that has made him the target of hostile coverage in the state-dominated media in those countries. 
        Hungarian government spokesman Zoltán Kovács said the “Visegrád Four“, which also includes Slovakia and the Czech Republic, had demonstrated their growing strength and influence over the direction of the EU, in part, because they had “toppled Timmermans“. 
        Kovács also boasted of the defeat of Manfred Weber, the centre-right candidate to become European commission president, who was the first choice of the German chancellor, Angela Merkel. 
        Socialist MEPs meeting in Strasbourg on Tuesday reacted with fury when reports emerged of the Von der Leyen compromise, largely because of the defeat for their candidate who has been the EU flag-bearer for the rule of law. 
        “It is unacceptable that populist governments represented in the council rule out the best candidate only because he has stood up for the rule of law and for our shared European values,“ said the Socialist MEP leader, Iratxe García. 
        """,
            language=Language(lang="en"),
            mentions=["ner", "wikipedia"],
            nbest=False,
            customisation="generic",
            min_selector_score=0.2,
        )
    )
    r = disambiguate.sync_detailed(client=client, multipart_data=form)
    if r.is_success:
        result: QueryResultText = r.parsed
        assert result is not None
        assert len(result.entities) > 0
        assert result.entities[0].raw_name == "Ursula von der Leyen"
        assert result.entities[0].wikidata_id == "Q60772"


def test_disambiguate_pdf():
    testdir = Path(__file__).parent / "data"
    # json_file = testdir / "scai_test_sherpa.json"
    pdf_file = testdir / "PMC1636350.pdf"
    with pdf_file.open("rb") as fin:
        client = Client(base_url="https://entityfishing.kairntech.com/service", timeout=300)
        form = DisambiguateForm(
            query=QueryParameters(
                language=Language(lang="en"),
                mentions=["wikipedia"],
                nbest=False,
                customisation="generic",
                min_selector_score=0.2,
                sentence=True,
                structure="grobid",
            ),
            file=File(file_name=pdf_file.name, payload=fin, mime_type="application/pdf"),
        )
        r = disambiguate.sync_detailed(client=client, multipart_data=form)
        if r.is_success:
            result: QueryResultFile = r.parsed
            assert result is not None
            assert len(result.entities) > 0
            assert len(result.pages) > 0
            assert len(result.entities[0].pos) > 0


def test_disambiguate_vector():
    client = Client(base_url="https://entityfishing.kairntech.com/service", timeout=300)
    form = DisambiguateForm(
        query=QueryParameters(
            #            term_vector=[WeightedTerm(term="Jaguar", score=1.0), WeightedTerm(term="car", score=1.0)],
            language=Language(lang="en"),
            mentions=["wikipedia"],
            nbest=False,
            customisation="generic",
            min_selector_score=0.2,
            sentence=False,
        )
    )
    r = disambiguate.sync_detailed(client=client, multipart_data=form)
    if r.is_success:
        result: QueryResultTermVector = r.parsed
        assert result is not None
        assert len(result.term_vector) > 0
        assert result.term_vector[0].entities[0].preferred_term == "Jaguar Cars"


def test_get_concept():
    client = Client(base_url="https://entityfishing.kairntech.com/service")
    r = get_concept.sync_detailed(id="Q60772", client=client)
    result: Concept = r.parsed
    if r.is_success:
        assert result is not None
        assert result.raw_name == "Ursula von der Leyen"
        assert result.wikidata_id == "Q60772"


def test_get_conceptb_error():
    client = Client(base_url="https://entityfishing.kairntech.com/service")
    r = get_concept.sync_detailed(id="XXXX", client=client)
    if r.is_success:
        pass
    else:
        assert r.status_code != 200
        with pytest.raises(HTTPError) as ex_info:
            r.raise_for_status()
        assert "Invalid" in str(ex_info.value)


def test_term_lookup():
    client = Client(base_url="https://entityfishing.kairntech.com/service")
    r = term_lookup.sync_detailed(term="Paris", lang="fr", client=client)
    result: TermSenses = r.parsed
    if r.is_success:
        assert result is not None
        assert result.lang == "fr"
        assert result.senses[0].preferred == "Paris"
        assert result.senses[0].pageid == 681159
        assert result.senses[0].prob_c > 0.9
