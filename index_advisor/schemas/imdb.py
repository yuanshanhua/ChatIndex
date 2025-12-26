from .schema import Column, DatabaseSchema, TableSchema, add_table_stats


imdb_schema = DatabaseSchema(
    "imdb",
    table_count=21,
    tables=[
        TableSchema(
            "aka_name",
            [
                Column(name="id", typ="integer"),
                Column(name="imdb_index", typ="character varying"),
                Column(name="md5sum", typ="character varying"),
                Column(name="name", typ="character varying"),
                Column(name="name_pcode_cf", typ="character varying"),
                Column(name="name_pcode_nf", typ="character varying"),
                Column(name="person_id", typ="integer"),
                Column(name="surname_pcode", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "aka_title",
            [
                Column(name="episode_nr", typ="integer"),
                Column(name="episode_of_id", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="imdb_index", typ="character varying"),
                Column(name="kind_id", typ="integer"),
                Column(name="md5sum", typ="character varying"),
                Column(name="movie_id", typ="integer"),
                Column(name="note", typ="character varying"),
                Column(name="phonetic_code", typ="character varying"),
                Column(name="production_year", typ="integer"),
                Column(name="season_nr", typ="integer"),
                Column(name="title", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "cast_info",
            [
                Column(name="id", typ="integer"),
                Column(name="movie_id", typ="integer"),
                Column(name="note", typ="character varying"),
                Column(name="nr_order", typ="integer"),
                Column(name="person_id", typ="integer"),
                Column(name="person_role_id", typ="integer"),
                Column(name="role_id", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "char_name",
            [
                Column(name="id", typ="integer"),
                Column(name="imdb_id", typ="integer"),
                Column(name="imdb_index", typ="character varying"),
                Column(name="md5sum", typ="character varying"),
                Column(name="name", typ="character varying"),
                Column(name="name_pcode_nf", typ="character varying"),
                Column(name="surname_pcode", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "comp_cast_type",
            [
                Column(name="id", typ="integer"),
                Column(name="kind", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "company_name",
            [
                Column(name="country_code", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="imdb_id", typ="integer"),
                Column(name="md5sum", typ="character varying"),
                Column(name="name", typ="character varying"),
                Column(name="name_pcode_nf", typ="character varying"),
                Column(name="name_pcode_sf", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "company_type",
            [
                Column(name="id", typ="integer"),
                Column(name="kind", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "complete_cast",
            [
                Column(name="id", typ="integer"),
                Column(name="movie_id", typ="integer"),
                Column(name="status_id", typ="integer"),
                Column(name="subject_id", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "info_type",
            [
                Column(name="id", typ="integer"),
                Column(name="info", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "keyword",
            [
                Column(name="id", typ="integer"),
                Column(name="keyword", typ="character varying"),
                Column(name="phonetic_code", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "kind_type",
            [
                Column(name="id", typ="integer"),
                Column(name="kind", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "link_type",
            [
                Column(name="id", typ="integer"),
                Column(name="link", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "movie_companies",
            [
                Column(name="company_id", typ="integer"),
                Column(name="company_type_id", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="movie_id", typ="integer"),
                Column(name="note", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "movie_info_idx",
            [
                Column(name="id", typ="integer"),
                Column(name="info", typ="character varying"),
                Column(name="info_type_id", typ="integer"),
                Column(name="movie_id", typ="integer"),
                Column(name="note", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "movie_keyword",
            [
                Column(name="id", typ="integer"),
                Column(name="keyword_id", typ="integer"),
                Column(name="movie_id", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "movie_link",
            [
                Column(name="id", typ="integer"),
                Column(name="link_type_id", typ="integer"),
                Column(name="linked_movie_id", typ="integer"),
                Column(name="movie_id", typ="integer"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "name",
            [
                Column(name="gender", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="imdb_id", typ="integer"),
                Column(name="imdb_index", typ="character varying"),
                Column(name="md5sum", typ="character varying"),
                Column(name="name", typ="character varying"),
                Column(name="name_pcode_cf", typ="character varying"),
                Column(name="name_pcode_nf", typ="character varying"),
                Column(name="surname_pcode", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "role_type",
            [
                Column(name="id", typ="integer"),
                Column(name="role", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "title",
            [
                Column(name="episode_nr", typ="integer"),
                Column(name="episode_of_id", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="imdb_id", typ="integer"),
                Column(name="imdb_index", typ="character varying"),
                Column(name="kind_id", typ="integer"),
                Column(name="md5sum", typ="character varying"),
                Column(name="phonetic_code", typ="character varying"),
                Column(name="production_year", typ="integer"),
                Column(name="season_nr", typ="integer"),
                Column(name="series_years", typ="character varying"),
                Column(name="title", typ="character varying"),
            ],
            primary_key="id",
            large_for_index=["title"],
        ),
        TableSchema(
            "movie_info",
            [
                Column(name="id", typ="integer"),
                Column(name="info", typ="character varying"),
                Column(name="info_type_id", typ="integer"),
                Column(name="movie_id", typ="integer"),
                Column(name="note", typ="character varying"),
            ],
            primary_key="id",
            large_for_index=["info"],
        ),
        TableSchema(
            "person_info",
            [
                Column(name="id", typ="integer"),
                Column(name="info", typ="character varying"),
                Column(name="info_type_id", typ="integer"),
                Column(name="note", typ="character varying"),
                Column(name="person_id", typ="integer"),
            ],
            primary_key="id",
            large_for_index=["info"],
        ),
    ],
    index_count=21,
    indexes=[],
)


# 为aka_name表添加外键
imdb_schema.add_foreign_key("aka_name", "person_id", "name", "id")

# 为aka_title表添加外键
imdb_schema.add_foreign_key("aka_title", "movie_id", "title", "id")
imdb_schema.add_foreign_key("aka_title", "kind_id", "kind_type", "id")
# imdb_schema.add_foreign_key("aka_title", "episode_of_id", "title", "id")

# 为cast_info表添加外键
imdb_schema.add_foreign_key("cast_info", "movie_id", "title", "id")
imdb_schema.add_foreign_key("cast_info", "person_id", "name", "id")
imdb_schema.add_foreign_key("cast_info", "person_role_id", "char_name", "id")
imdb_schema.add_foreign_key("cast_info", "role_id", "role_type", "id")

# 为complete_cast表添加外键
imdb_schema.add_foreign_key("complete_cast", "movie_id", "title", "id")
imdb_schema.add_foreign_key("complete_cast", "status_id", "comp_cast_type", "id")
imdb_schema.add_foreign_key("complete_cast", "subject_id", "comp_cast_type", "id")

# 为movie_companies表添加外键
imdb_schema.add_foreign_key("movie_companies", "movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_companies", "company_id", "company_name", "id")
imdb_schema.add_foreign_key("movie_companies", "company_type_id", "company_type", "id")

# 为movie_info表添加外键
imdb_schema.add_foreign_key("movie_info", "movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_info", "info_type_id", "info_type", "id")

# 为movie_info_idx表添加外键
imdb_schema.add_foreign_key("movie_info_idx", "movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_info_idx", "info_type_id", "info_type", "id")

# 为movie_keyword表添加外键
imdb_schema.add_foreign_key("movie_keyword", "movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_keyword", "keyword_id", "keyword", "id")

# 为movie_link表添加外键
imdb_schema.add_foreign_key("movie_link", "movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_link", "linked_movie_id", "title", "id")
imdb_schema.add_foreign_key("movie_link", "link_type_id", "link_type", "id")

# 为person_info表添加外键
imdb_schema.add_foreign_key("person_info", "person_id", "name", "id")
imdb_schema.add_foreign_key("person_info", "info_type_id", "info_type", "id")

# 为title表添加外键
imdb_schema.add_foreign_key("title", "kind_id", "kind_type", "id")
imdb_schema.add_foreign_key("title", "episode_of_id", "title", "id")

# 预计算表统计信息
add_table_stats(
    imdb_schema,
    "comp_cast_type",
    4,
    {
        "id": 4,
        "kind": 4,
    },
)
add_table_stats(
    imdb_schema,
    "aka_title",
    361472,
    {
        "md5sum": 346289,
        "id": 361472,
        "movie_id": 205631,
        "kind_id": 6,
        "production_year": 128,
        "episode_nr": 162,
        "episode_of_id": 754,
        "phonetic_code": 20013,
        "season_nr": 40,
        "title": 313624,
        "note": 3360,
        "imdb_index": 12,
    },
)
add_table_stats(
    imdb_schema,
    "company_name",
    234997,
    {
        "md5sum": 234997,
        "id": 234997,
        "country_code": 215,
        "name": 224384,
        "name_pcode_sf": 17446,
        "name_pcode_nf": 18486,
        "imdb_id": 0,
    },
)
add_table_stats(imdb_schema, "company_type", 4, {"id": 4, "kind": 4})
add_table_stats(
    imdb_schema,
    "complete_cast",
    135086,
    {
        "status_id": 2,
        "subject_id": 2,
        "id": 135086,
        "movie_id": 93514,
    },
)
add_table_stats(imdb_schema, "info_type", 113, {"info": 113, "id": 113})
add_table_stats(imdb_schema, "kind_type", 7, {"id": 7, "kind": 7})
add_table_stats(imdb_schema, "link_type", 18, {"id": 18, "link": 18})
add_table_stats(
    imdb_schema,
    "keyword",
    134170,
    {
        "keyword": 134170,
        "id": 134170,
        "phonetic_code": 15481,
    },
)
add_table_stats(
    imdb_schema,
    "aka_name",
    901343,
    {
        "md5sum": 810625,
        "person_id": 588222,
        "id": 901343,
        "name": 810619,
        "name_pcode_cf": 22017,
        "name_pcode_nf": 21138,
        "imdb_index": 5,
        "surname_pcode": 4312,
    },
)
add_table_stats(
    imdb_schema,
    "movie_link",
    29997,
    {
        "id": 29997,
        "linked_movie_id": 16169,
        "link_type_id": 16,
        "movie_id": 6411,
    },
)
add_table_stats(
    imdb_schema,
    "movie_info_idx",
    1380035,
    {
        "id": 1380035,
        "movie_id": 459925,
        "info": 146245,
        "info_type_id": 5,
        "note": 0,
    },
)
add_table_stats(
    imdb_schema,
    "char_name",
    3140339,
    {
        "md5sum": 3140339,
        "id": 3140339,
        "name": 3139742,
        "name_pcode_nf": 23076,
        "imdb_id": 0,
        "imdb_index": 6,
        "surname_pcode": 17198,
    },
)
add_table_stats(
    imdb_schema,
    "movie_companies",
    2609129,
    {
        "id": 2609129,
        "movie_id": 1087236,
        "company_id": 234997,
        "company_type_id": 2,
        "note": 66450,
    },
)
add_table_stats(
    imdb_schema,
    "movie_keyword",
    4523930,
    {
        "id": 4523930,
        "movie_id": 476794,
        "keyword_id": 134170,
    },
)
add_table_stats(imdb_schema, "role_type", 12, {"role": 12, "id": 12})
add_table_stats(
    imdb_schema,
    "movie_info",
    14835711,
    {
        "id": 14835711,
        "movie_id": 2468825,
        "info": 2720920,
        "info_type_id": 71,
        "note": 133601,
    },
)
add_table_stats(
    imdb_schema,
    "name",
    4167491,
    {
        "md5sum": 4167491,
        "id": 4167491,
        "gender": 2,
        "name": 3587400,
        "name_pcode_cf": 23603,
        "name_pcode_nf": 23261,
        "imdb_id": 0,
        "imdb_index": 179,
        "surname_pcode": 4671,
    },
)
add_table_stats(
    imdb_schema,
    "person_info",
    2963632,
    {
        "person_id": 550717,
        "id": 2963632,
        "info": 1925044,
        "info_type_id": 22,
        "note": 49988,
    },
)
add_table_stats(
    imdb_schema,
    "title",
    2528312,
    {
        "md5sum": 2528312,
        "id": 2528312,
        "series_years": 1407,
        "kind_id": 6,
        "production_year": 132,
        "episode_nr": 14906,
        "episode_of_id": 51481,
        "phonetic_code": 23259,
        "title": 1483631,
        "season_nr": 96,
        "imdb_id": 0,
        "imdb_index": 24,
    },
)
add_table_stats(
    imdb_schema,
    "cast_info",
    36244344,
    {
        "person_id": 4051810,
        "nr_order": 1094,
        "role_id": 11,
        "id": 36244344,
        "movie_id": 2331601,
        "person_role_id": 3140339,
        "note": 715571,
    },
)
