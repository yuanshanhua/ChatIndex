from .schema import Column, DatabaseSchema, TableSchema


gifshow_schema = DatabaseSchema(
    "gifshow",
    table_count=11,
    tables=[
        TableSchema(
            "ad_dsp_account_mirror",
            [
                Column(name="id", typ="bigint"),
                Column(name="put_status", typ="bigint"),
                Column(name="review_status", typ="bigint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "ad_dsp_campaign_mirror",
            [
                Column(name="id", typ="bigint"),
                Column(name="put_status", typ="bigint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "mirror_ad_dsp_creative_for_search",
            [
                Column(name="photo_id", typ="bigint"),
                Column(name="put_status", typ="bigint"),
                Column(name="review_status", typ="bigint"),
                Column(name="search_score_status", typ="bigint"),
                Column(name="unit_id", typ="bigint"),
            ],
        ),
        TableSchema(
            "mirror_ad_dsp_live_stream_user_info_for_search",
            [
                Column(name="live_stream_id", typ="bigint"),
                Column(name="user_id", typ="bigint"),
            ],
        ),
        TableSchema(
            "mirror_ad_dsp_unit_for_search",
            [
                Column(name="id", typ="bigint"),
                Column(name="item_id", typ="bigint"),
                Column(name="live_user_id", typ="bigint"),
                Column(name="put_status", typ="bigint"),
                Column(name="review_status", typ="bigint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "mirror_item_id2_set_result",
            [
                Column(name="item_id", typ="bigint"),
                Column(name="set_result", typ="bigint"),
            ],
        ),
        TableSchema(
            "wt_auto_bid_account",
            [
                Column(name="account_id", typ="bigint"),
                Column(name="table_id", typ="numeric"),
            ],
        ),
        TableSchema(
            "wt_auto_bid_campaign",
            [
                Column(name="campaign_id", typ="bigint"),
                Column(name="table_id", typ="numeric"),
            ],
        ),
        TableSchema(
            "wt_live",
            [
                Column(name="id", typ="bigint"),
                Column(name="live_stream_id", typ="bigint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "wt_product",
            [
                Column(name="id", typ="bigint"),
                Column(name="product_id", typ="bigint"),
                Column(name="spu_id_v1", typ="bigint"),
            ],
            primary_key="id",
        ),
        TableSchema(
            "wt_roi_live_data",
            [
                Column(name="campaign_id", typ="bigint"),
                Column(name="creative_id", typ="bigint"),
                Column(name="unit_id", typ="bigint"),
                Column(name="user_id", typ="bigint"),
            ],
        ),
    ],
    index_count=5,
    indexes=[],
)
