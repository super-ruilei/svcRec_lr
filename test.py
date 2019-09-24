# coding= utf-8
# 本程序处理协同过滤和推荐的问题
# 1.读取原始数据
# 2.格式化结构数据
# 3.处理稀疏矩阵的问题(dataset1中没有涉及到稀疏矩阵) 需要生成偏好矩阵,
# 3.1 随机生成一个用户-web服务的稀疏矩阵，因为有两个表格，因此应当生成两个稀疏矩阵，分别为rtMatrix_sprase.txt 和tpMatrix_sprise.txt。
# 3.1 根据偏好矩阵进行推荐系统的推荐运算实验。在实验数据得到后，对比参照精确度的时候使用完整的rtMatrix.txt
# 4.寻找用户之间的欧几里得距离（euclidea nmetric）（也称欧式距离）
# 5.计算用户表的皮尔逊相关度
# 6.计算Webservice的欧式区里相似度
# 7.计算webservice的皮尔逊相关度
# 8.使用基于用户的协同过滤(User-CF)输出推荐结果
# 8.使用基于物品的协同过滤(Item-CF)输出推荐结果
# refrence: http://python.jobbole.com/86563/
# refrence: https://github.com/zthero/python/blob/master/collective/code/test/make_data.py

import sys, os, random, json
import pickle
import math
from uitls import saveObj, getGeoDistance, loadObjsIfExist
import scipy
from scipy import *

##原始数据
RAW_USRLIST = "./dataset/userlist.txt"
RAW_WSLIST = "./dataset/wslist.txt"

##参照数据
RAW_TPMATRIX = "./dataset/tpMatrix.txt"
RAW_RTMATRIX = "./dataset/rtMatrix.txt"

##生成的偏好矩阵
GEN_TPMATRIX = "./dataset/tpMatrix.pref.txt"
GEN_RTMATRIX = "./dataset/rtMatrix.pref.txt"


def WSUser(ID="0", IPAddr="0.0.0.0", Country="", IPNo="0", AS="0", Latitude="0", Longitude="0"):

    result = {}
    result['ID'] = ID
    result['IPAddr'] = IPAddr
    result['Country'] = Country
    result['IPNo'] = IPNo
    result['AS'] = AS
    result['Latitude'] = Latitude
    result['Longitude'] = Longitude
    return result


def WSService(ID="0", WSDLAddress="", ServiceProvider="", IPAddr="0.0.0.0", Country="", IPNo="0", AS="0", Latitude="0",
              Longitude="0"):
    result = {}
    result['ID'] = ID
    # result['WSDLAddress']=WSDLAddress
    # result['ServiceProvider']=ServiceProvider
    result['IPAddr'] = IPAddr
    # result['Country']=Country
    result['IPNo'] = IPNo
    # result['AS']=AS
    result['Latitude'] = Latitude
    result['Longitude'] = Longitude
    return result


# 用户字典
UserList = {}
# web服务字典
WsList = {}

# 生成的偏好矩阵
Pref_TPMatrix = {}
Pref_RTMatrix = {}

# 原始的访问数据，用来做对比
TPMatrix = []
RTMatrix = []


# 读取用户列表
def readUserList():
    uFile = open(RAW_USRLIST, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        if (len(lns) < 6):
            print("Line: ", line, "has problem, skip")
            continue
        newUsr = WSUser(lns[0], lns[1], lns[2], lns[3], lns[4], lns[5], lns[6])
        UserList[lns[0]] = newUsr
    print("total read User:", len(UserList))


# 读取web服务列表
def readWsList():
    uFile = open(RAW_WSLIST, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        if (len(lns) < 8):
            print("Line: ", line, "has problem, skip")
            continue
        newWs = WSService(lns[0], lns[1], lns[2], lns[3], lns[4], lns[5], lns[6], lns[7], lns[8])
        WsList[lns[0]] = newWs
    print("total read Webservices:", len(WsList))


# 读取响应时间关系矩阵
def readTPandRX():
    uFile = open(RAW_TPMATRIX, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        TPMatrix.append(lns)
    print("total read user-item matrix of response-time.:", len(TPMatrix), "*", len(TPMatrix[0]))
    uFile = open(RAW_RTMATRIX, "r")
    for line in uFile:
        if (line.startswith("[") or line.startswith("=")):
            continue
        # print line
        lns = line.split('\t')
        RTMatrix.append(lns)
    print("total read user-item matrix for throughput.:", len(RTMatrix), "*", len(RTMatrix[0]))


# 生成一个偏好矩阵
## 583,*1165,1747,2330,2912
def generate_PrefMat():
    ram = random.SystemRandom()
    for u1 in UserList.keys():
        usr = UserList[u1]
        # 随机选取一些(0-1000个之间)web服务的ID然后查找出响应时间即可。
        rn = int(ram.random() * 1000)
        rn = 2912
        ln = len(WsList.keys())
        webs = {}
        for r1 in range(rn):
            # 随机选一个web服务
            l1 = int(ram.random() * ln)
            ws = WsList[str(l1)]
            # 查找出对应的响应时间属性
            rt = RTMatrix[int(usr['ID'])][int(ws['ID'])]
            webs[ws['ID']] = float(rt)
        Pref_RTMatrix[usr['ID']] = webs
        print(usr['ID'], usr['IPAddr'], len(webs))
    saveObj(Pref_RTMatrix, GEN_RTMATRIX)
    print(len(Pref_RTMatrix))


# 计算基于地理位置的相关度
def sim_geo(prefs, p1, p2):
    si = {}
    for itemId in prefs[p1]:
        if itemId in prefs[p2]:
            si[itemId] = 1
    # no same item
    if len(si) == 0: return 0
    SLAT1 = UserList[p1]['Latitude']
    SLON1 = UserList[p1]['Longitude']
    SLAT2 = UserList[p2]['Latitude']
    SLON2 = UserList[p2]['Longitude']
    if SLAT1.startswith("null") or SLON1.startswith("null") or SLAT2.startswith("null") or SLON2.startswith("null"):
        return 0
    LAT1 = float(SLAT1)
    LON1 = float(SLON1)
    LAT2 = float(SLAT2)
    LON2 = float(SLON2)
    # print LAT1,LON1,LAT2,LON2
    try:
        distance = getGeoDistance(LAT1, LON1, LAT2, LON2)
    except ZeroDivisionError:
        ##WTF happened?
        return 0
    except ValueError:
        return 0

    return 1000 / (distance + 1)


# 欧几里得距离
def sim_distance(prefs, p1, p2):
    si = {}
    for itemId in prefs[p1]:
        if itemId in prefs[p2]:
            si[itemId] = 1
    # no same item
    if len(si) == 0: return 0
    sum_of_squares = 0.0

    # 计算距离
    sum_of_squares = sum([pow(prefs[p1][item] - prefs[p2][item], 2) for item in si])
    return 1 / (1 + math.sqrt(sum_of_squares))
    pass


# 皮尔逊相关度
# 皮尔逊相关系数是一种度量两个变量间相关程度的方法。它是一个介于 1 和 -1 之间的值，其中，1 表示变量完全正相关， 0 表示无关，-1 表示完全负相关。
def sim_pearson(prefs, p1, p2):
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]: si[item] = 1
    if len(si) == 0: return 0
    n = len(si)
    # 计算开始
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    num = pSum - (sum1 * sum2 / n)
    den = math.sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    # 计算结束
    if den == 0: return 0
    r = num / den
    return r


# 推荐用户
def topMatches(prefs, person, n=5, similarity=sim_distance):
    # python列表推导式
    scores = [(similarity(prefs, person, other), other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


# 基于用户推荐物品
def getRecommendations(prefs, person, similarity=sim_pearson):
    totals = {}
    simSums = {}

    for other in prefs:
        # 不和自己做比较
        if other == person:
            continue
        sim = similarity(prefs, person, other)
        # 去除负相关的用户
        if sim == 0: continue
        for item in prefs[other]:
            # 只对自己没见过的服务做评价
            if item in prefs[person]: continue
            totals.setdefault(item, 0)
            totals[item] += sim * prefs[other][item]
            simSums.setdefault(item, 0)
            simSums[item] += sim
    # 归一化处理生成推荐列表
    rankings = []
    for item in totals:
        if simSums[item] is not 0:
            rankings.append((totals[item] / simSums[item], item))
    # rankings=[(total/simSums[item],item) for item,total in totals.items()]
    rankings.sort()
    rankings.reverse()
    return rankings


# 基于物品的列表
def transformPrefs(prefs):
    itemList = {}
    for person in prefs:
        for item in prefs[person]:
            if not itemList.has_key(item):
                itemList[item] = {}
                # result.setdefault(item,{})
            itemList[item][person] = prefs[person][item]
    return itemList


# 构建基于物品相似度数据集
def calculateSimilarItems(prefs, n=10):
    result = {}
    itemPrefs = transformPrefs(prefs)
    c = 0
    for item in itemPrefs:
        c += 1
        if c % 10 == 0: print("%d / %d" % (c, len(itemPrefs)))
        scores = topMatches(itemPrefs, item, n=n, similarity=sim_distance)
        result[item] = scores
    return result


# 基于物品的推荐
def getRecommendedItems(prefs, itemMatch, user):
    userRatings = prefs[user]
    scores = {}
    totalSim = {}
    # Loop over items rated by this user
    for (item, rating) in userRatings.items():
        # Loop over items similar to this one
        for (similarity, item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2, 0)
            scores[item2] += similarity * rating
            # Sum of all the similarities
            totalSim.setdefault(item2, 0)
            totalSim[item2] += similarity

    # Divide each total score by total weighting to get an average
    rankings = [(score / totalSim[item], item) for item, score in scores.items()]

    # Return the rankings from highest to lowest
    rankings.sort()
    rankings.reverse()
    return rankings


###API
def getRecommend(UserID, sim):
    Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    similarity = sim_pearson
    if sim == "distance":
        similarity = sim_distance
    if sim == "geo":
        similarity = sim_geo

    recomm = getRecommendations(Pref_RTMatrix, UserID, similarity)
    result = {}
    result["UserInfo"] = WsList[UserID]
    result["Recommend Services"] = []
    for im in recomm:
        # 取大于2.0的推荐值，这个值可以自己定义，如果取值太小，则表现出推荐太多无用的内容。
        if im[0] > 2.0:
            result["Recommend Services"].append(WsList[im[1]])
            # print im , WsList[im[1]]
        else:
            continue
    result["Total"] = len(result["Recommend Services"])
    print(sim, result["Total"], result["UserInfo"])
    jresult = json.dumps(result)
    return jresult


def getUserListJson():
    result = json.dumps(UserList)
    return result


def getWebServiceListJson():
    result = json.dumps(WsList)
    return result


def getAll():
    for usr in UserList.keys():
        getRecommend(usr, "pearson")
    for usr in UserList.keys():
        getRecommend(usr, "distance")
    for usr in UserList.keys():
        getRecommend(usr, "geo")

def getAccuracy(target,result):
    accu = 0
    top_eculiean_100 = []
    for ii in range(100):
        top_eculiean_100.append(int(result[ii][1]))
    print(top_eculiean_100)

    return top_eculiean_100



def mainRun():
    readUserList()
    readWsList()
    readTPandRX()
    Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    if (Pref_RTMatrix is None):
        generate_PrefMat()
        ## just reload PrefMatrix
        Pref_RTMatrix = loadObjsIfExist(GEN_RTMATRIX)
    # print sim_distance(Pref_RTMatrix, "101","20")
    # print sim_pearson(Pref_RTMatrix,"101","20")
    result_eculidean =  getRecommendations(Pref_RTMatrix,"1",sim_distance)
    result_pearson = getRecommendations(Pref_RTMatrix, "1", sim_pearson)
    result_geo = getRecommendations(Pref_RTMatrix, "1", sim_geo)

    accuray_eculidean = getAccuracy("eculidean", result_eculidean)
    accuray_pearson = getAccuracy("pearson", result_pearson)
    accuray_geo = getAccuracy( "geo",result_geo)

    # print len(result_eculidean)
    # for im in result_eculidean[0:40]:
    #    print im , WsList[im[1]]
    # if len(sys.argv) > 1:
    #     getAll()
    #

mainRun()