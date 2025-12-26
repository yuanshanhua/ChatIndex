from .schema import Column, DatabaseSchema, TableSchema


fnhk_schema = DatabaseSchema(
    "fnhk",
    table_count=3,
    tables=[
        TableSchema(
            "pripady",
            [
                Column(name="datum_prijeti", typ="character varying"),
                Column(name="datum_propusteni", typ="character varying"),
                Column(name="delka_hospitalizace", typ="integer"),
                Column(name="drg_skupina", typ="integer"),
                Column(name="identifikace_pripadu", typ="integer"),
                Column(name="identifikator_pacienta", typ="integer"),
                Column(name="kod_zdravotni_pojistovny", typ="integer"),
                Column(name="pohlavi_pacienta", typ="character"),
                Column(name="psc", typ="character"),
                Column(name="seznam_vedlejsich_diagnoz", typ="character varying"),
                Column(name="vekovy_interval_pacienta", typ="character varying"),
                Column(name="zakladni_diagnoza", typ="character varying"),
            ],
            primary_key="identifikace_pripadu",
        ),
        TableSchema(
            "vykony",
            [
                Column(name="body", typ="integer"),
                Column(name="datum_provedeni_vykonu", typ="character varying"),
                Column(name="identifikace_pripadu", typ="integer"),
                Column(name="kod_polozky", typ="integer"),
                Column(name="pocet", typ="integer"),
                Column(name="typ_polozky", typ="integer"),
            ],
            primary_key=["identifikace_pripadu", "datum_provedeni_vykonu", "kod_polozky"],
        ),
        TableSchema(
            "zup",
            [
                Column(name="cena", typ="numeric"),
                Column(name="datum_provedeni_vykonu", typ="character varying"),
                Column(name="identifikace_pripadu", typ="integer"),
                Column(name="kod_polozky", typ="integer"),
                Column(name="pocet", typ="numeric"),
                Column(name="typ_polozky", typ="integer"),
            ],
            primary_key=["identifikace_pripadu", "datum_provedeni_vykonu", "kod_polozky"],
        ),
    ],
    index_count=3,
    indexes=[],
)
