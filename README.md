# entifyfishing-client
A client library for accessing [Entity-fishing](https://github.com/kermitt2/entity-fishing) - Entity Recognition and Disambiguation

## Usage
First, create a client:

```python
from entifyfishing_client import Client

client = Client(base_url="http://nerd.huma-num.fr/nerd/service")
```

Now call your endpoint and use your models:

```python
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
    
r = get_concept.sync_detailed(id="Q40", client=client)
result: Concept = r.parsed
if r.is_success:
    assert result is not None
    assert result.raw_name == "Austria"
    assert result.wikidata_id == "Q40"    
    assert len(result.statements) > 0
```

Or do the same thing with an async version:

```python
pdf_file = "MyPDFFile.pdf"
with pdf_file.open("rb") as fin:
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
    r = await disambiguate.asyncio_detailed(client=client, multipart_data=form)
    if r.is_success:
        result: QueryResultFile = r.parsed
        assert result is not None
        assert len(result.entities) > 0
        assert len(result.pages) > 0
        assert len(result.entities[0].pos) > 0
```

By default, when you're calling an HTTPS API it will attempt to verify that SSL is working correctly. Using certificate verification is highly recommended most of the time, but sometimes you may need to authenticate to a server (especially an internal server) using a custom certificate bundle.

```python
client = Client(
    base_url="http://nerd.huma-num.fr/nerd/service", 
    verify_ssl="/path/to/certificate_bundle.pem",
)
```

You can also disable certificate validation altogether, but beware that **this is a security risk**.

```python
client = Client(
    verify_ssl=False
)
```

Things to know:
1. Every path/method combo becomes a Python module with four functions:
    1. `sync`: Blocking request that returns parsed data (if successful) or `None`
    1. `sync_detailed`: Blocking request that always returns a `Request`, optionally with `parsed` set if the request was successful.
    1. `asyncio`: Like `sync` but the async instead of blocking
    1. `asyncio_detailed`: Like `sync_detailed` by async instead of blocking

1. All path/query params, and bodies become method arguments.
1. If your endpoint had any tags on it, the first tag will be used as a module name for the function (my_tag above)
1. Any endpoint which did not have a tag will be in `entifyfishing_client.api.default`

## Building / publishing this Client
This project uses [Poetry](https://python-poetry.org/) to manage dependencies  and packaging.  Here are the basics:
1. Update the metadata in pyproject.toml (e.g. authors, version)
1. If you're using a private repository, configure it with Poetry
    1. `poetry config repositories.<your-repository-name> <url-to-your-repository>`
    1. `poetry config http-basic.<your-repository-name> <username> <password>`
1. Publish the client with `poetry publish --build -r <your-repository-name>` or, if for public PyPI, just `poetry publish --build`

If you want to install this client into another project without publishing it (e.g. for development) then:
1. If that project **is using Poetry**, you can simply do `poetry add <path-to-this-client>` from that project
1. If that project is not using Poetry:
    1. Build a wheel with `poetry build -f wheel`
    1. Install that wheel from the other project `pip install <path-to-wheel>`
