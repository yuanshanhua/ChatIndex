from .schema import Column, DatabaseSchema, TableSchema


grants_schema = DatabaseSchema(
    "grants",
    table_count=12,
    tables=[
        TableSchema(
            "awards",
            [
                Column(name="abstract_narration", typ="character varying"),
                Column(name="arra_amount", typ="integer"),
                Column(name="award_amount", typ="integer"),
                Column(name="award_effective_date", typ="character varying"),
                Column(name="award_expiration_date", typ="character varying"),
                Column(name="award_id", typ="integer"),
                Column(name="award_instrument", typ="character varying"),
                Column(name="award_title", typ="character varying"),
                Column(name="max_amd_letter_date", typ="character varying"),
                Column(name="min_amd_letter_date", typ="character varying"),
                Column(name="organisation_code", typ="integer"),
                Column(name="program_officer", typ="character varying"),
            ],
            primary_key="award_id",
            large_for_index=["abstract_narration"],
        ),
        TableSchema(
            "foa_info",
            [
                Column(name="code", typ="integer"),
                Column(name="name", typ="character varying"),
            ],
            primary_key="code",
        ),
        TableSchema(
            "foa_info_awards",
            [
                Column(name="award_id", typ="integer"),
                Column(name="code", typ="integer"),
            ],
            primary_key=["award_id", "code"],
        ),
        TableSchema(
            "institution",
            [
                Column(name="address", typ="character varying"),
                Column(name="city_name", typ="character varying"),
                Column(name="contact", typ="integer"),
                Column(name="country_name", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="name", typ="character varying"),
                Column(name="state_code", typ="character varying"),
                Column(name="state_name", typ="character varying"),
                Column(name="zipcode", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "institution_awards",
            [
                Column(name="award_id", typ="integer"),
                Column(name="id", typ="integer"),
                Column(name="name", typ="character varying"),
                Column(name="zipcode", typ="character varying"),
            ],
            primary_key=["award_id", "id"],
        ),
        TableSchema(
            "investigator",
            [
                Column(name="email_id", typ="character varying"),
                Column(name="first_name", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="last_name", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "investigator_awards",
            [
                Column(name="award_id", typ="integer"),
                Column(name="email_id", typ="integer"),
                Column(name="end_date", typ="character varying"),
                Column(name="role_code", typ="character varying"),
                Column(name="start_date", typ="character varying"),
            ],
            primary_key=["award_id", "email_id"],
        ),
        TableSchema(
            "organization",
            [
                Column(name="code", typ="integer"),
                Column(name="directorate", typ="character varying"),
                Column(name="division", typ="character varying"),
            ],
            primary_key="code",
        ),
        TableSchema(
            "program_element",
            [
                Column(name="code", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="text", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "program_element_awards",
            [
                Column(name="award_id", typ="integer"),
                Column(name="code", typ="integer"),
            ],
            primary_key=["award_id", "code"],
        ),
        TableSchema(
            "program_reference",
            [
                Column(name="code", typ="character varying"),
                Column(name="id", typ="integer"),
                Column(name="text", typ="character varying"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "program_reference_awards",
            [
                Column(name="award_id", typ="integer"),
                Column(name="code", typ="integer"),
            ],
            primary_key=["award_id", "code"],
        ),
    ],
    index_count=12,
    indexes=[],
)
