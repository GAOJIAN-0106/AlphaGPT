"""
EDB indicator → futures product mapping.

Maps TianQin EDB indicator IDs to the futures product addresses
used in DuckDB (format: "{product_id}.{exchange}").

Each entry: product_address → list of EDB IDs (primary first).
When multiple IDs exist for the same product, the first is preferred.
"""

# Warehouse receipt (仓单) data: EDB ID → product address
# Only includes products that have active futures contracts
WAREHOUSE_EDB_MAP = {
    # === SHFE ===
    'cu.SHFE': [179],           # 铜:总计
    'al.SHFE': [181],           # 铝:总计
    'zn.SHFE': [187],           # 锌:总计
    'pb.SHFE': [203],           # 铅:总计
    'ni.SHFE': [188],           # 镍:总计
    'sn.SHFE': [191],           # 锡:总计
    'ag.SHFE': [189],           # 白银:总计
    'au.SHFE': [208],           # 黄金:总计
    'rb.SHFE': [182],           # 螺纹钢
    'hc.SHFE': [204],           # 热轧卷板
    'ss.SHFE': [216],           # 不锈钢
    'wr.SHFE': [451],           # 线材
    'sp.SHFE': [217],           # 纸浆:小计
    'bu.SHFE': [443],           # 沥青:仓库:总计
    'ru.SHFE': [207],           # 天然橡胶:总计
    'fu.SHFE': [466],           # 燃料油:总计
    'ao.SHFE': [206],           # 氧化铝:总计
    # === DCE ===
    'i.DCE': [201],             # 铁矿石
    'j.DCE': [205],             # 焦炭
    'jm.DCE': [190],            # 焦煤
    'c.DCE': [186],             # 玉米
    'cs.DCE': [452],            # 玉米淀粉
    'm.DCE': [183],             # 豆粕
    'y.DCE': [199],             # 豆油
    'a.DCE': [215],             # 豆一
    'b.DCE': [440],             # 豆二
    'p.DCE': [213],             # 棕榈油
    'l.DCE': [444],             # 聚乙烯:总计
    'pp.DCE': [449],            # 聚丙烯:总计
    'v.DCE': [438],             # 聚氯乙烯:总计
    'eg.DCE': [442],            # 乙二醇:总计
    'eb.DCE': [448],            # 苯乙烯:总计
    'pg.DCE': [462],            # 液化石油气:总计
    'lh.DCE': [194],            # 生猪
    'jd.DCE': [211],            # 鸡蛋
    'rr.DCE': [461],            # 粳米
    # === CZCE ===
    'TA.CZCE': [196],           # PTA:总计
    'MA.CZCE': [214],           # 甲醇:总计
    'SA.CZCE': [185],           # 纯碱:总计
    'FG.CZCE': [192],           # 玻璃:总计
    'SR.CZCE': [209],           # 白糖
    'CF.CZCE': [197],           # 一号棉
    'OI.CZCE': [437],           # 菜籽油
    'RM.CZCE': [195],           # 菜粕
    'UR.CZCE': [439],           # 尿素:总计
    'SF.CZCE': [202],           # 硅铁
    'SM.CZCE': [198],           # 锰硅
    'AP.CZCE': [441],           # 苹果
    'CJ.CZCE': [200],           # 红枣
    'PK.CZCE': [218],           # 花生
    'PF.CZCE': [445],           # 短纤:总计
    'CY.CZCE': [454],           # 棉纱
    'SH.CZCE': [212],           # 烧碱:总计
    'PX.CZCE': [458],           # 对二甲苯:总计
    # === INE ===
    'sc.INE': [447],            # 中质含硫原油
    'lu.INE': [463],            # 低硫燃料油
    'nr.INE': [210],            # 20号胶:总计
    'bc.INE': [30969],          # 铜(BC):总计
    # === GFEX ===
    'si.GFEX': [184],           # 工业硅:小计
    'lc.GFEX': [180],           # 碳酸锂:小计
}

# Social inventory (期货库存/社会库存) data: weekly frequency
INVENTORY_EDB_MAP = {
    'rb.SHFE': [13],            # 期货库存:螺纹钢:小计
    'hc.SHFE': [16],            # 期货库存:热轧卷板:小计
    'wr.SHFE': [15],            # 期货库存:线材:小计
    'cu.SHFE': [24],            # 期货库存:铜
    'al.SHFE': [26],            # 期货库存:铝
    'zn.SHFE': [29],            # 期货库存:锌
    'pb.SHFE': [32],            # 期货库存:铅
    'ni.SHFE': [31],            # 期货库存:镍
    'ag.SHFE': [23],            # 期货库存:白银
    'au.SHFE': [25],            # 期货库存:黄金
    'sp.SHFE': [17],            # 期货库存:纸浆:小计
    'ao.SHFE': [27],            # 期货库存:氧化铝
    'i.DCE': [28],              # 期货库存:铁矿石
    'bc.INE': [30],             # 期货库存:铜(BC)
}
