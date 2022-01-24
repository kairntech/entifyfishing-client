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
    WeightedTerm,
)
from entifyfishing_client.types import File


def test_disambiguate_text():
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service", timeout=300)
    form = DisambiguateForm(
        query=QueryParameters(
            text="""Austria invaded and fought the Serbian army at the Battle of Cer and Battle of Kolubara beginning on 12 August. 
            The army, led by general Paul von Hindenburg defeated Russia in a series of battles collectively known as the First Battle of Tannenberg (17 August – 2 September). 
            But the failed Russian invasion, causing the fresh German troops to move to the east, allowed the tactical Allied victory at the First Battle of the Marne. 
            Unfortunately for the Allies, the pro-German King Constantine I dismissed the pro-Allied government of E. Venizelos before the Allied expeditionary force could arrive.
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
        assert result.entities[0].raw_name == "Austria"
        assert result.entities[0].wikidata_id == "Q40"


def test_disambiguate_pdf():
    testdir = Path(__file__).parent / "data"
    # json_file = testdir / "scai_test_sherpa.json"
    pdf_file = testdir / "PMC1636350.pdf"
    with pdf_file.open("rb") as fin:
        client = Client(base_url="http://nerd.huma-num.fr/nerd//service", timeout=300)
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
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service", timeout=300)
    form = DisambiguateForm(
        query=QueryParameters(
            term_vector=[WeightedTerm(term="Jaguar", score=1.0), WeightedTerm(term="car", score=1.0)],
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
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service")
    r = get_concept.sync_detailed(id="Q60772", client=client)
    result: Concept = r.parsed
    if r.is_success:
        assert result is not None
        assert result.raw_name == "Ursula von der Leyen"
        assert result.wikidata_id == "Q60772"


def test_get_concept_wiki():
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service")
    r = get_concept.sync_detailed(id="438549", lang="fr", client=client)
    result: Concept = r.parsed
    if r.is_success:
        assert result is not None
        assert result.raw_name == "Ursula von der Leyen"
        assert result.wikidata_id == "Q60772"

def test_get_conceptb_error():
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service")
    r = get_concept.sync_detailed(id="XXXX", client=client)
    if r.is_success:
        pass
    else:
        assert r.status_code != 200
        with pytest.raises(HTTPError) as ex_info:
            r.raise_for_status()
        assert "Invalid" in str(ex_info.value)


def test_term_lookup():
    client = Client(base_url="http://nerd.huma-num.fr/nerd//service")
    r = term_lookup.sync_detailed(term="Paris", lang="fr", client=client)
    result: TermSenses = r.parsed
    if r.is_success:
        assert result is not None
        assert result.lang == "fr"
        assert result.senses[0].preferred == "Paris"
        assert result.senses[0].pageid == 681159
        assert result.senses[0].prob_c > 0.9
