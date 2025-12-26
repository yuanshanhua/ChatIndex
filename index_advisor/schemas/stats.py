from .schema import Column, DatabaseSchema, TableSchema


stats_schema = DatabaseSchema(
    "stats",
    table_count=8,
    tables=[
        TableSchema(
            "badges",
            [
                Column(name="date", typ="bigint"),
                Column(name="id", typ="integer"),
                Column(name="userid", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "comments",
            [
                Column(name="creationdate", typ="bigint"),
                Column(name="id", typ="integer"),
                Column(name="postid", typ="integer"),
                Column(name="score", typ="smallint"),
                Column(name="userid", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "posthistory",
            [
                Column(name="creationdate", typ="bigint"),
                Column(name="id", typ="integer"),
                Column(name="posthistorytypeid", typ="smallint"),
                Column(name="postid", typ="integer"),
                Column(name="userid", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "postlinks",
            [
                Column(name="creationdate", typ="bigint"),
                Column(name="id", typ="integer"),
                Column(name="linktypeid", typ="smallint"),
                Column(name="postid", typ="integer"),
                Column(name="relatedpostid", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "posts",
            [
                Column(name="answercount", typ="integer"),
                Column(name="commentcount", typ="integer"),
                Column(name="creationdate", typ="bigint"),
                Column(name="favoritecount", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="lasteditoruserid", typ="integer"),
                Column(name="owneruserid", typ="integer"),
                Column(name="posttypeid", typ="smallint"),
                Column(name="score", typ="integer"),
                Column(name="viewcount", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "tags",
            [
                Column(name="count", typ="integer"),
                Column(name="excerptpostid", typ="integer"),
                Column(name="id", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "users",
            [
                Column(name="creationdate", typ="bigint"),
                Column(name="downvotes", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="reputation", typ="integer"),
                Column(name="upvotes", typ="integer"),
                Column(name="views", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "votes",
            [
                Column(name="bountyamount", typ="smallint"),
                Column(name="creationdate", typ="bigint"),
                Column(name="id", typ="integer"),
                Column(name="postid", typ="integer"),
                Column(name="userid", typ="integer"),
                Column(name="votetypeid", typ="smallint"),
            ],
            primary_key="id",
        ),
    ],
    index_count=8,
    indexes=[],
)
