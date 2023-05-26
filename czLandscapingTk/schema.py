# Auto generated from czLandscaping.yaml by pythongen.py version: 0.9.0
# Generation date: 2023-05-26T12:49:43
# Schema: czLandscaping
#
# id: https://chanzuckerberg.github.io/czLandscapingTk/linkml/czLandscaping
# description: LinkML Schema for scientific landscaping analysis performed at CZI.
# license: https://creativecommons.org/publicdomain/zero/1.0/

import dataclasses
import sys
import re
from jsonasobj2 import JsonObj, as_dict
from typing import Optional, List, Union, Dict, ClassVar, Any
from dataclasses import dataclass
from linkml_runtime.linkml_model.meta import EnumDefinition, PermissibleValue, PvFormulaOptions

from linkml_runtime.utils.slot import Slot
from linkml_runtime.utils.metamodelcore import empty_list, empty_dict, bnode
from linkml_runtime.utils.yamlutils import YAMLRoot, extended_str, extended_float, extended_int
from linkml_runtime.utils.dataclass_extensions_376 import dataclasses_init_fn_with_kwargs
from linkml_runtime.utils.formatutils import camelcase, underscore, sfx
from linkml_runtime.utils.enumerations import EnumDefinitionImpl
from rdflib import Namespace, URIRef
from linkml_runtime.utils.curienamespace import CurieNamespace
from linkml_runtime.linkml_model.types import Date, Integer, String, Uriorcurie
from linkml_runtime.utils.metamodelcore import URIorCURIE, XSDDate

metamodel_version = "1.7.0"
version = None

# Overwrite dataclasses _init_fn to add **kwargs in __init__
dataclasses._init_fn = dataclasses_init_fn_with_kwargs

# Namespaces
MESH = CurieNamespace('MESH', 'http://id.nlm.nih.gov/mesh/')
ORCID = CurieNamespace('ORCID', 'https://orcid.org/')
WIKIDATA = CurieNamespace('WIKIDATA', 'https://www.wikidata.org/entity/')
WIKIDATA_PROPERTY = CurieNamespace('WIKIDATA_PROPERTY', 'https://www.wikidata.org/prop/')
BIOLINK = CurieNamespace('biolink', 'https://w3id.org/biolink/vocab/')
CZLSA = CurieNamespace('czlsa', 'https://chanzuckerberg.github.io/czLandscaping/')
DCT = CurieNamespace('dct', 'http://purl.org/dc/terms/')
DOI = CurieNamespace('doi', 'http://example.org/UNKNOWN/doi/')
FABIO = CurieNamespace('fabio', 'http://purl.org/spar/fabio/')
IAO = CurieNamespace('iao', 'http://purl.obolibrary.org/obo/IAO_')
LINKML = CurieNamespace('linkml', 'https://w3id.org/linkml/')
RDF = CurieNamespace('rdf', 'http://www.w3.org/1999/02/22-rdf-syntax-ns#')
RDFS = CurieNamespace('rdfs', 'http://www.w3.org/2000/01/rdf-schema#')
SCHEMA = CurieNamespace('schema', 'http://schema.org/')
DEFAULT_ = CZLSA


# Types

# Class references
class EntityId(extended_str):
    pass


class NamedThingId(EntityId):
    pass


class InformationContentEntityId(NamedThingId):
    pass


class WorkId(InformationContentEntityId):
    pass


class InformationResourceId(InformationContentEntityId):
    pass


class WorkCollectionId(InformationContentEntityId):
    pass


class WorkFragmentId(InformationContentEntityId):
    pass


class SelectorId(InformationContentEntityId):
    pass


class OffsetTextSelectorId(SelectorId):
    pass


class PersonId(NamedThingId):
    pass


class AuthorId(PersonId):
    pass


class OrganizationId(NamedThingId):
    pass


class CityId(NamedThingId):
    pass


class CountryId(NamedThingId):
    pass


@dataclass
class Entity(YAMLRoot):
    """
    Root Model class for all things and informational relationships, real or imagined.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Entity
    class_class_curie: ClassVar[str] = "czlsa:Entity"
    class_name: ClassVar[str] = "Entity"
    class_model_uri: ClassVar[URIRef] = CZLSA.Entity

    id: Union[str, EntityId] = None
    iri: Optional[str] = None
    type: Optional[Union[str, List[str]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, EntityId):
            self.id = EntityId(self.id)

        if self.iri is not None and not isinstance(self.iri, str):
            self.iri = str(self.iri)

        if not isinstance(self.type, list):
            self.type = [self.type] if self.type is not None else []
        self.type = [v if isinstance(v, str) else str(v) for v in self.type]

        super().__post_init__(**kwargs)


@dataclass
class NamedThing(Entity):
    """
    an entity or concept/class described by a name
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.NamedThing
    class_class_curie: ClassVar[str] = "czlsa:NamedThing"
    class_name: ClassVar[str] = "NamedThing"
    class_model_uri: ClassVar[URIRef] = CZLSA.NamedThing

    id: Union[str, NamedThingId] = None
    name: Optional[str] = None
    xref: Optional[Union[Union[str, URIorCURIE], List[Union[str, URIorCURIE]]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if not isinstance(self.xref, list):
            self.xref = [self.xref] if self.xref is not None else []
        self.xref = [v if isinstance(v, URIorCURIE) else URIorCURIE(v) for v in self.xref]

        super().__post_init__(**kwargs)


@dataclass
class InformationContentEntity(NamedThing):
    """
    a piece of information that typically describes some topic of discourse or is used as support.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.InformationContentEntity
    class_class_curie: ClassVar[str] = "czlsa:InformationContentEntity"
    class_name: ClassVar[str] = "InformationContentEntity"
    class_model_uri: ClassVar[URIRef] = CZLSA.InformationContentEntity

    id: Union[str, InformationContentEntityId] = None
    license: Optional[str] = None
    rights: Optional[str] = None
    format: Optional[str] = None
    creation_date: Optional[Union[str, XSDDate]] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self.license is not None and not isinstance(self.license, str):
            self.license = str(self.license)

        if self.rights is not None and not isinstance(self.rights, str):
            self.rights = str(self.rights)

        if self.format is not None and not isinstance(self.format, str):
            self.format = str(self.format)

        if self.creation_date is not None and not isinstance(self.creation_date, XSDDate):
            self.creation_date = XSDDate(self.creation_date)

        super().__post_init__(**kwargs)


@dataclass
class Work(InformationContentEntity):
    """
    A published
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Work
    class_class_curie: ClassVar[str] = "czlsa:Work"
    class_name: ClassVar[str] = "Work"
    class_model_uri: ClassVar[URIRef] = CZLSA.Work

    id: Union[str, WorkId] = None
    has_part: Optional[Union[Union[str, WorkFragmentId], List[Union[str, WorkFragmentId]]]] = empty_list()
    authors: Optional[Union[Dict[Union[str, AuthorId], Union[dict, "Author"]], List[Union[dict, "Author"]]]] = empty_dict()
    title: Optional[str] = None
    abstract: Optional[str] = None
    full_text: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, WorkId):
            self.id = WorkId(self.id)

        if not isinstance(self.has_part, list):
            self.has_part = [self.has_part] if self.has_part is not None else []
        self.has_part = [v if isinstance(v, WorkFragmentId) else WorkFragmentId(v) for v in self.has_part]

        self._normalize_inlined_as_list(slot_name="authors", slot_type=Author, key_name="id", keyed=True)

        if self.title is not None and not isinstance(self.title, str):
            self.title = str(self.title)

        if self.abstract is not None and not isinstance(self.abstract, str):
            self.abstract = str(self.abstract)

        if self.full_text is not None and not isinstance(self.full_text, str):
            self.full_text = str(self.full_text)

        super().__post_init__(**kwargs)


@dataclass
class InformationResource(InformationContentEntity):
    """
    A database or knowledgebase and its supporting ecosystem of interfaces and services that deliver content to
    consumers (e.g. web portals, APIs, query endpoints, streaming services, data downloads, etc.). A single
    Information Resource by this definition may span many different datasets or databases, and include many access
    endpoints and user interfaces. Information Resources include project-specific resources such as a Translator
    Knowledge Provider, and community knowledgebases like ChemBL, OMIM, or DGIdb.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.InformationResource
    class_class_curie: ClassVar[str] = "czlsa:InformationResource"
    class_name: ClassVar[str] = "InformationResource"
    class_model_uri: ClassVar[URIRef] = CZLSA.InformationResource

    id: Union[str, InformationResourceId] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, InformationResourceId):
            self.id = InformationResourceId(self.id)

        super().__post_init__(**kwargs)


@dataclass
class WorkCollection(InformationContentEntity):
    """
    A collection of works.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.WorkCollection
    class_class_curie: ClassVar[str] = "czlsa:WorkCollection"
    class_name: ClassVar[str] = "WorkCollection"
    class_model_uri: ClassVar[URIRef] = CZLSA.WorkCollection

    id: Union[str, WorkCollectionId] = None
    name: Optional[str] = None
    logical_query: Optional[str] = None
    creation_date: Optional[Union[str, XSDDate]] = None
    information_sources: Optional[Union[Union[str, InformationResourceId], List[Union[str, InformationResourceId]]]] = empty_list()
    has_part: Optional[Union[Union[str, WorkId], List[Union[str, WorkId]]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, WorkCollectionId):
            self.id = WorkCollectionId(self.id)

        if self.name is not None and not isinstance(self.name, str):
            self.name = str(self.name)

        if self.logical_query is not None and not isinstance(self.logical_query, str):
            self.logical_query = str(self.logical_query)

        if self.creation_date is not None and not isinstance(self.creation_date, XSDDate):
            self.creation_date = XSDDate(self.creation_date)

        if not isinstance(self.information_sources, list):
            self.information_sources = [self.information_sources] if self.information_sources is not None else []
        self.information_sources = [v if isinstance(v, InformationResourceId) else InformationResourceId(v) for v in self.information_sources]

        if not isinstance(self.has_part, list):
            self.has_part = [self.has_part] if self.has_part is not None else []
        self.has_part = [v if isinstance(v, WorkId) else WorkId(v) for v in self.has_part]

        super().__post_init__(**kwargs)


@dataclass
class WorkFragment(InformationContentEntity):
    """
    A selected subportion of the contents of a Work, described by an selector.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.WorkFragment
    class_class_curie: ClassVar[str] = "czlsa:WorkFragment"
    class_name: ClassVar[str] = "WorkFragment"
    class_model_uri: ClassVar[URIRef] = CZLSA.WorkFragment

    id: Union[str, WorkFragmentId] = None
    part_of: Optional[Union[str, WorkId]] = None
    selector: Optional[Union[str, SelectorId]] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, WorkFragmentId):
            self.id = WorkFragmentId(self.id)

        if self.part_of is not None and not isinstance(self.part_of, WorkId):
            self.part_of = WorkId(self.part_of)

        if self.selector is not None and not isinstance(self.selector, SelectorId):
            self.selector = SelectorId(self.selector)

        super().__post_init__(**kwargs)


@dataclass
class Selector(InformationContentEntity):
    """
    A way of localizing and describing a WorkFragment within a Work.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Selector
    class_class_curie: ClassVar[str] = "czlsa:Selector"
    class_name: ClassVar[str] = "Selector"
    class_model_uri: ClassVar[URIRef] = CZLSA.Selector

    id: Union[str, SelectorId] = None

@dataclass
class OffsetTextSelector(Selector):
    """
    A way of localizing and describing a fragment of text within a larger body of text using offsets and lengths.
    """
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.OffsetTextSelector
    class_class_curie: ClassVar[str] = "czlsa:OffsetTextSelector"
    class_name: ClassVar[str] = "OffsetTextSelector"
    class_model_uri: ClassVar[URIRef] = CZLSA.OffsetTextSelector

    id: Union[str, OffsetTextSelectorId] = None
    offset: Optional[int] = None
    length: Optional[int] = None
    text: Optional[str] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, OffsetTextSelectorId):
            self.id = OffsetTextSelectorId(self.id)

        if self.offset is not None and not isinstance(self.offset, int):
            self.offset = int(self.offset)

        if self.length is not None and not isinstance(self.length, int):
            self.length = int(self.length)

        if self.text is not None and not isinstance(self.text, str):
            self.text = str(self.text)

        super().__post_init__(**kwargs)


@dataclass
class Person(NamedThing):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Person
    class_class_curie: ClassVar[str] = "czlsa:Person"
    class_name: ClassVar[str] = "Person"
    class_model_uri: ClassVar[URIRef] = CZLSA.Person

    id: Union[str, PersonId] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, PersonId):
            self.id = PersonId(self.id)

        super().__post_init__(**kwargs)


@dataclass
class Author(Person):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Author
    class_class_curie: ClassVar[str] = "czlsa:Author"
    class_name: ClassVar[str] = "Author"
    class_model_uri: ClassVar[URIRef] = CZLSA.Author

    id: Union[str, AuthorId] = None
    orcid: Optional[str] = None
    affiliations: Optional[Union[Union[str, OrganizationId], List[Union[str, OrganizationId]]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, AuthorId):
            self.id = AuthorId(self.id)

        if self.orcid is not None and not isinstance(self.orcid, str):
            self.orcid = str(self.orcid)

        if not isinstance(self.affiliations, list):
            self.affiliations = [self.affiliations] if self.affiliations is not None else []
        self.affiliations = [v if isinstance(v, OrganizationId) else OrganizationId(v) for v in self.affiliations]

        super().__post_init__(**kwargs)


@dataclass
class Organization(NamedThing):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Organization
    class_class_curie: ClassVar[str] = "czlsa:Organization"
    class_name: ClassVar[str] = "Organization"
    class_model_uri: ClassVar[URIRef] = CZLSA.Organization

    id: Union[str, OrganizationId] = None
    city: Optional[Union[Union[str, CityId], List[Union[str, CityId]]]] = empty_list()
    country: Optional[Union[Union[str, CountryId], List[Union[str, CountryId]]]] = empty_list()

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, OrganizationId):
            self.id = OrganizationId(self.id)

        if not isinstance(self.city, list):
            self.city = [self.city] if self.city is not None else []
        self.city = [v if isinstance(v, CityId) else CityId(v) for v in self.city]

        if not isinstance(self.country, list):
            self.country = [self.country] if self.country is not None else []
        self.country = [v if isinstance(v, CountryId) else CountryId(v) for v in self.country]

        super().__post_init__(**kwargs)


@dataclass
class City(NamedThing):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.City
    class_class_curie: ClassVar[str] = "czlsa:City"
    class_name: ClassVar[str] = "City"
    class_model_uri: ClassVar[URIRef] = CZLSA.City

    id: Union[str, CityId] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, CityId):
            self.id = CityId(self.id)

        super().__post_init__(**kwargs)


@dataclass
class Country(NamedThing):
    _inherited_slots: ClassVar[List[str]] = []

    class_class_uri: ClassVar[URIRef] = CZLSA.Country
    class_class_curie: ClassVar[str] = "czlsa:Country"
    class_name: ClassVar[str] = "Country"
    class_model_uri: ClassVar[URIRef] = CZLSA.Country

    id: Union[str, CountryId] = None

    def __post_init__(self, *_: List[str], **kwargs: Dict[str, Any]):
        if self._is_empty(self.id):
            self.MissingRequiredField("id")
        if not isinstance(self.id, CountryId):
            self.id = CountryId(self.id)

        super().__post_init__(**kwargs)


# Enumerations


# Slots
class slots:
    pass

slots.id = Slot(uri=CZLSA.id, name="id", curie=CZLSA.curie('id'),
                   model_uri=CZLSA.id, domain=Entity, range=Union[str, EntityId])

slots.orcid = Slot(uri=CZLSA.orcid, name="orcid", curie=CZLSA.curie('orcid'),
                   model_uri=CZLSA.orcid, domain=None, range=Optional[str])

slots.iri = Slot(uri=CZLSA.iri, name="iri", curie=CZLSA.curie('iri'),
                   model_uri=CZLSA.iri, domain=None, range=Optional[str])

slots.type = Slot(uri=RDF.type, name="type", curie=RDF.curie('type'),
                   model_uri=CZLSA.type, domain=Entity, range=Optional[Union[str, List[str]]])

slots.name = Slot(uri=RDFS.label, name="name", curie=RDFS.curie('label'),
                   model_uri=CZLSA.name, domain=Entity, range=Optional[str])

slots.xref = Slot(uri=CZLSA.xref, name="xref", curie=CZLSA.curie('xref'),
                   model_uri=CZLSA.xref, domain=NamedThing, range=Optional[Union[Union[str, URIorCURIE], List[Union[str, URIorCURIE]]]])

slots.license = Slot(uri=CZLSA.license, name="license", curie=CZLSA.curie('license'),
                   model_uri=CZLSA.license, domain=InformationContentEntity, range=Optional[str])

slots.rights = Slot(uri=CZLSA.rights, name="rights", curie=CZLSA.curie('rights'),
                   model_uri=CZLSA.rights, domain=InformationContentEntity, range=Optional[str])

slots.format = Slot(uri=CZLSA.format, name="format", curie=CZLSA.curie('format'),
                   model_uri=CZLSA.format, domain=InformationContentEntity, range=Optional[str])

slots.creation_date = Slot(uri=CZLSA.creation_date, name="creation date", curie=CZLSA.curie('creation_date'),
                   model_uri=CZLSA.creation_date, domain=None, range=Optional[Union[str, XSDDate]])

slots.has_part = Slot(uri=CZLSA.has_part, name="has part", curie=CZLSA.curie('has_part'),
                   model_uri=CZLSA.has_part, domain=None, range=Optional[Union[str, WorkFragmentId]])

slots.part_of = Slot(uri=CZLSA.part_of, name="part of", curie=CZLSA.curie('part_of'),
                   model_uri=CZLSA.part_of, domain=None, range=Optional[Union[str, WorkId]])

slots.logical_query = Slot(uri=CZLSA.logical_query, name="logical query", curie=CZLSA.curie('logical_query'),
                   model_uri=CZLSA.logical_query, domain=WorkCollection, range=Optional[str])

slots.authors = Slot(uri=CZLSA.authors, name="authors", curie=CZLSA.curie('authors'),
                   model_uri=CZLSA.authors, domain=Work, range=Optional[Union[Dict[Union[str, AuthorId], Union[dict, "Author"]], List[Union[dict, "Author"]]]])

slots.title = Slot(uri=CZLSA.title, name="title", curie=CZLSA.curie('title'),
                   model_uri=CZLSA.title, domain=Work, range=Optional[str])

slots.abstract = Slot(uri=CZLSA.abstract, name="abstract", curie=CZLSA.curie('abstract'),
                   model_uri=CZLSA.abstract, domain=Work, range=Optional[str])

slots.full_text = Slot(uri=CZLSA.full_text, name="full text", curie=CZLSA.curie('full_text'),
                   model_uri=CZLSA.full_text, domain=Work, range=Optional[str])

slots.cites = Slot(uri=CZLSA.cites, name="cites", curie=CZLSA.curie('cites'),
                   model_uri=CZLSA.cites, domain=Work, range=Optional[Union[Union[str, WorkId], List[Union[str, WorkId]]]])

slots.information_sources = Slot(uri=CZLSA.information_sources, name="information sources", curie=CZLSA.curie('information_sources'),
                   model_uri=CZLSA.information_sources, domain=WorkCollection, range=Optional[Union[Union[str, InformationResourceId], List[Union[str, InformationResourceId]]]])

slots.selector = Slot(uri=CZLSA.selector, name="selector", curie=CZLSA.curie('selector'),
                   model_uri=CZLSA.selector, domain=WorkFragment, range=Optional[Union[str, SelectorId]])

slots.affiliations = Slot(uri=CZLSA.affiliations, name="affiliations", curie=CZLSA.curie('affiliations'),
                   model_uri=CZLSA.affiliations, domain=Author, range=Optional[Union[Union[str, OrganizationId], List[Union[str, OrganizationId]]]])

slots.offsetTextSelector__offset = Slot(uri=CZLSA.offset, name="offsetTextSelector__offset", curie=CZLSA.curie('offset'),
                   model_uri=CZLSA.offsetTextSelector__offset, domain=None, range=Optional[int])

slots.offsetTextSelector__length = Slot(uri=CZLSA.length, name="offsetTextSelector__length", curie=CZLSA.curie('length'),
                   model_uri=CZLSA.offsetTextSelector__length, domain=None, range=Optional[int])

slots.offsetTextSelector__text = Slot(uri=CZLSA.text, name="offsetTextSelector__text", curie=CZLSA.curie('text'),
                   model_uri=CZLSA.offsetTextSelector__text, domain=None, range=Optional[str])

slots.organization__city = Slot(uri=CZLSA.city, name="organization__city", curie=CZLSA.curie('city'),
                   model_uri=CZLSA.organization__city, domain=None, range=Optional[Union[Union[str, CityId], List[Union[str, CityId]]]])

slots.organization__country = Slot(uri=CZLSA.country, name="organization__country", curie=CZLSA.curie('country'),
                   model_uri=CZLSA.organization__country, domain=None, range=Optional[Union[Union[str, CountryId], List[Union[str, CountryId]]]])

slots.Work_has_part = Slot(uri=CZLSA.has_part, name="Work_has part", curie=CZLSA.curie('has_part'),
                   model_uri=CZLSA.Work_has_part, domain=Work, range=Optional[Union[Union[str, WorkFragmentId], List[Union[str, WorkFragmentId]]]])

slots.WorkCollection_has_part = Slot(uri=CZLSA.has_part, name="WorkCollection_has part", curie=CZLSA.curie('has_part'),
                   model_uri=CZLSA.WorkCollection_has_part, domain=WorkCollection, range=Optional[Union[Union[str, WorkId], List[Union[str, WorkId]]]])

slots.WorkFragment_part_of = Slot(uri=CZLSA.part_of, name="WorkFragment_part of", curie=CZLSA.curie('part_of'),
                   model_uri=CZLSA.WorkFragment_part_of, domain=WorkFragment, range=Optional[Union[str, WorkId]])
