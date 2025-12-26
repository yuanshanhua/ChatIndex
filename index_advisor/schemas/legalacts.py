from .schema import Column, DatabaseSchema, TableSchema


legalacts_schema = DatabaseSchema(
    "legalacts",
    table_count=5,
    tables=[
        TableSchema(
            "legalact_link",
            [
                Column(name="actid1", typ="integer"),
                Column(name="actid2", typ="integer"),
            ],
            primary_key=["actid1", "actid2"],
        ),
        TableSchema(
            "legalact_people",
            [
                Column(name="actid", typ="integer"),
                Column(name="peopleid", typ="integer"),
            ],
            primary_key=["peopleid", "actid"],
        ),
        TableSchema(
            "legalacts",
            [
                Column(name="actkind", typ="character varying"),
                Column(name="actlink", typ="integer"),
                Column(name="actnumber", typ="smallint"),
                Column(name="actyear", typ="smallint"),
                Column(name="casekind", typ="character varying"),
                Column(name="casenumber", typ="smallint"),
                Column(name="court", typ="character varying"),
                Column(name="hash", typ="character"),
                Column(name="highcourt", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="judge", typ="character varying"),
                Column(name="legaldate", typ="date"),
                Column(name="motivedate", typ="date"),
                Column(name="motivelink", typ="integer"),
                Column(name="outnumber", typ="smallint"),
                Column(name="resultofappeal", typ="character varying"),
                Column(name="senddate", typ="date"),
                Column(name="startdate", typ="date"),
                Column(name="status", typ="character varying"),
                Column(name="typeofdocument", typ="character varying"),
                Column(name="update", typ="timestamp without time zone"),
                Column(name="yearhighercourt", typ="smallint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "people",
            [
                Column(name="court", typ="character varying"),
                Column(name="jury", typ="integer"),
                Column(name="name", typ="character varying"),
                Column(name="personid", typ="integer"),
            ],
            primary_key="personid",
        ),
        TableSchema(
            "scrapefix",
            [
                Column(name="actid", typ="integer"),
                Column(name="contributor", typ="character varying"),
                Column(name="fix_description", typ="text"),
            ],
            primary_key="actid",
            large_for_index=["fix_description"],
        ),
    ],
    index_count=5,
    indexes=[],
)
