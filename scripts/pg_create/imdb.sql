drop database if exists imdb;
create database imdb;
\c imdb;
drop table if exists aka_name;
create table aka_name (
    id integer not null,
    person_id integer not null,
    name character varying,
    imdb_index varchar(3),
    name_pcode_cf varchar(11),
    name_pcode_nf varchar(11),
    surname_pcode varchar(11),
    md5sum varchar(65),
    primary key (id)
);
drop table if exists aka_title;
create table aka_title (
    id integer not null,
    movie_id integer not null,
    title character varying,
    imdb_index varchar(4),
    kind_id integer not null,
    production_year integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note varchar(72),
    md5sum varchar(32),
    primary key (id)
);
drop table if exists cast_info;
create table cast_info (
    id integer not null,
    person_id integer not null,
    movie_id integer not null,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer not null,
    primary key (id)
);
drop table if exists char_name;
create table char_name (
    id integer not null,
    name character varying,
    imdb_index varchar(2),
    imdb_id integer,
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32),
    primary key (id)
);
drop table if exists comp_cast_type;
create table comp_cast_type (
    id integer not null,
    kind varchar(32) not null,
    primary key (id)
);
drop table if exists company_name;
create table company_name (
    id integer not null,
    name character varying,
    country_code varchar(6),
    imdb_id integer,
    name_pcode_nf varchar(5),
    name_pcode_sf varchar(5),
    md5sum varchar(32),
    primary key (id)
);
drop table if exists company_type;
create table company_type (
    id integer not null,
    kind varchar(32),
    primary key (id)
);
drop table if exists complete_cast;
create table complete_cast (
    id integer not null,
    movie_id integer,
    subject_id integer not null,
    status_id integer not null,
    primary key (id)
);
drop table if exists info_type;
create table info_type (
    id integer not null,
    info varchar(32) not null,
    primary key (id)
);
drop table if exists keyword;
create table keyword (
    id integer not null,
    keyword character varying not null,
    phonetic_code varchar(5),
    primary key (id)
);
drop table if exists kind_type;
create table kind_type (
    id integer not null,
    kind varchar(15),
    primary key (id)
);
drop table if exists link_type;
create table link_type (
    id integer not null,
    link varchar(32) not null,
    primary key (id)
);
drop table if exists movie_companies;
create table movie_companies (
    id integer not null,
    movie_id integer not null,
    company_id integer not null,
    company_type_id integer not null,
    note character varying,
    primary key (id)
);
drop table if exists movie_info_idx;
create table movie_info_idx (
    id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying not null,
    note varchar(1),
    primary key (id)
);
drop table if exists movie_keyword;
create table movie_keyword (
    id integer not null,
    movie_id integer not null,
    keyword_id integer not null,
    primary key (id)
);
drop table if exists movie_link;
create table movie_link (
    id integer not null,
    movie_id integer not null,
    linked_movie_id integer not null,
    link_type_id integer not null,
    primary key (id)
);
drop table if exists name;
create table name (
    id integer not null,
    name character varying not null,
    imdb_index varchar(9),
    imdb_id integer,
    gender varchar(1),
    name_pcode_cf varchar(5),
    name_pcode_nf varchar(5),
    surname_pcode varchar(5),
    md5sum varchar(32),
    primary key (id)
);
drop table if exists role_type;
create table role_type (
    id integer not null,
    role varchar(32) not null,
    primary key (id)
);
drop table if exists title;
create table title (
    id integer not null,
    title character varying,
    imdb_index varchar(5),
    kind_id integer not null,
    production_year integer,
    imdb_id integer,
    phonetic_code varchar(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years varchar(49),
    md5sum varchar(32),
    primary key (id)
);
drop table if exists movie_info;
create table movie_info (
    id integer not null,
    movie_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying,
    primary key (id)
);
drop table if exists person_info;
create table person_info (
    id integer not null,
    person_id integer not null,
    info_type_id integer not null,
    info character varying,
    note character varying,
    primary key (id)
);
create index idx_movie_keyword_movie_id on movie_keyword using btree(movie_id);
create index idx_movie_companies_company_id on movie_companies using btree(company_id);
create index idx_movie_companies_company_type_id on movie_companies using btree(company_type_id);
create index idx_movie_info_idx_movie_id on movie_info_idx using btree(movie_id);
create index idx_aka_name_person_id on aka_name using btree(person_id);
create index idx_movie_info_idx_info_type_id on movie_info_idx using btree(info_type_id);
create index idx_person_info_person_id on person_info using btree(person_id);
create index idx_title_kind_id on title using btree(kind_id);
create index idx_cast_info_movie_id on cast_info using btree(movie_id);
create index idx_movie_companies_movie_id on movie_companies using btree(movie_id);
create index idx_cast_info_person_id on cast_info using btree(person_id);
create index idx_cast_info_person_role_id on cast_info using btree(person_role_id);
create index idx_movie_keyword_keyword_id on movie_keyword using btree(keyword_id);
create index idx_movie_info_movie_id on movie_info using btree(movie_id);
create index idx_aka_title_movie_id on aka_title using btree(movie_id);
create index idx_complete_cast_movie_id on complete_cast using btree(movie_id);
create index idx_cast_info_role_id on cast_info using btree(role_id);
\copy person_info from './person_info.csv' csv delimiter '|' null 'NULL' header;
analyze person_info;
\copy info_type from './info_type.csv' csv delimiter '|' null 'NULL' header;
analyze info_type;
\copy aka_title from './aka_title.csv' csv delimiter '|' null 'NULL' header;
analyze aka_title;
\copy name from './name.csv' csv delimiter '|' null 'NULL' header;
analyze name;
\copy movie_info_idx from './movie_info_idx.csv' csv delimiter '|' null 'NULL' header;
analyze movie_info_idx;
\copy char_name from './char_name.csv' csv delimiter '|' null 'NULL' header;
analyze char_name;
\copy company_name from './company_name.csv' csv delimiter '|' null 'NULL' header;
analyze company_name;
\copy aka_name from './aka_name.csv' csv delimiter '|' null 'NULL' header;
analyze aka_name;
\copy cast_info from './cast_info.csv' csv delimiter '|' null 'NULL' header;
analyze cast_info;
\copy comp_cast_type from './comp_cast_type.csv' csv delimiter '|' null 'NULL' header;
analyze comp_cast_type;
\copy company_type from './company_type.csv' csv delimiter '|' null 'NULL' header;
analyze company_type;
\copy movie_link from './movie_link.csv' csv delimiter '|' null 'NULL' header;
analyze movie_link;
\copy keyword from './keyword.csv' csv delimiter '|' null 'NULL' header;
analyze keyword;
\copy title from './title.csv' csv delimiter '|' null 'NULL' header;
analyze title;
\copy movie_info from './movie_info.csv' csv delimiter '|' null 'NULL' header;
analyze movie_info;
\copy movie_keyword from './movie_keyword.csv' csv delimiter '|' null 'NULL' header;
analyze movie_keyword;
\copy link_type from './link_type.csv' csv delimiter '|' null 'NULL' header;
analyze link_type;
\copy complete_cast from './complete_cast.csv' csv delimiter '|' null 'NULL' header;
analyze complete_cast;
\copy kind_type from './kind_type.csv' csv delimiter '|' null 'NULL' header;
analyze kind_type;
\copy role_type from './role_type.csv' csv delimiter '|' null 'NULL' header;
analyze role_type;
\copy movie_companies from './movie_companies.csv' csv delimiter '|' null 'NULL' header;
analyze movie_companies;