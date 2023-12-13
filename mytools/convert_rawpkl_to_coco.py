import os.path as osp
import mmcv
from mmengine.fileio import dump, load
from mmengine.utils import track_iter_progress
from glob import glob
import os
import logging
import fire
import pickle
from multiprocessing import Pool
# convert pkl to coco json
ConvertLabelTypeDict = {
    "neg" : 0,
    "circle": 1,
    "up": 2,
    "down": 3,
    "left": 4,
    "right": 5,
    "u_turn": 6,
    "pedestrian": 7,
    "trafficlightPedestrian": 7,
    "CountdownLight": 8,
    "unknown": 9,
    "trafficlightBicycl": 10,
    "bicycle": 10,
    "noTrafficLight": 11,
    "FP": 12,
    "side":13,
    "digit":8,
    "trafficlightCircle":1,
    "trafficlightUp":2,
    "trafficlightDown":3,
    "trafficlightLeft":4,
    "trafficlightRight":5,
    "notClear": 9,
    "【0】": 8,
    "【1】": 8,
    "【2】": 8,
    "【3】": 8,
    "【4】": 8,
    "【5】": 8,
    "【6】": 8,
    "【7】": 8,
    "【8】": 8,
    "【9】": 8,
}

def processfile(inputpklfilefiles, datatype):
    currentinfo = []    
    obj_count = 0 
    img_count = 0
    logger.info('processing {} files'.format(len(inputpklfilefiles)))
    for inputpklfile in inputpklfilefiles:
        data_infos = pickle.load(open(inputpklfile, 'rb'))    
        for idx, v in enumerate(data_infos):
            img_path = v['cropimage']
            relativebox = v['relativebox']
            label = v['label']
            localfile = v['localfile']            
            height, width = mmcv.imread(img_path).shape[:2]            
            imageinfo = dict(id = idx , file_name = img_path, height=height, width=width)
            img_count += 1
            currentannotations = []
            if label is not None:
                if label not in ConvertLabelTypeDict:
                    # logger.error('label: {} not in ConvertLabelTypeDict'.format(label))
                    continue
                for bbox in relativebox:
                    x_min, y_min, x_max, y_max = bbox
                    data_anno = dict(
                        image_id=idx,
                        id=obj_count,
                        category_id=ConvertLabelTypeDict[label],
                        bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                        area=(x_max - x_min) * (y_max - y_min),
                        segmentation=[],
                        iscrowd=0)
                    currentannotations.append(data_anno)
                    obj_count += 1  
            else:
                data_anno = dict(
                    image_id=idx,
                    id=0,
                    category_id=0,
                    bbox=[],
                    area=0,
                    segmentation=[],
                    iscrowd=0)
                currentannotations.append(data_anno)
            currentinfo.append([imageinfo, currentannotations])
    logger.info('processed {} files'.format(len(inputpklfilefiles)))
    return currentinfo, obj_count, img_count, datatype

def processfilehelper(args):
    return processfile(*args)

def convert_rawpkl_to_coco_parallel(inputpklfiletxt, outputfolder):
    datefiledict = {}
    imageprefix = '/data-algorithm-hl/xianglong.meng/data/refine/sample/image/'
    filecount = 0
    dateset = set()
    trycount = -1
    with open(inputpklfiletxt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            inputpklfile = line.strip()
            image_prefix = os.path.dirname(inputpklfile)
            inputpklbasename = os.path.basename(inputpklfile)
            if 'LJ' not in inputpklfile:
                continue
            datestr = inputpklfile.split('/')[-3]
            if datestr not in datefiledict:
                datefiledict[datestr] = []
            datefiledict[datestr].append(inputpklfile)            
            filecount += 1
            dateset.add(datestr)
            if trycount > 0 and filecount > trycount:
                break
    logger.info('dateset : {}'.format(dateset))    
    logger.info('datefiledict: {}'.format(datefiledict)) 
    datelist = list(dateset)
    traindates = set(datelist[:int(len(datelist)*0.8)])
    logger.info('traindates: {}, {}'.format(traindates, len(datelist)))
    obj_count = 1
    args_list = []
    for datestr, inputpklfiles in track_iter_progress(list(datefiledict.items())):
        datatype = 'test'
        if datestr in traindates:
            datatype = 'train'
        args_list.append((inputpklfiles, datatype))
    num_processes = 8
    # 使用进程池处理转换
    totalresultdict = {}
    with Pool(num_processes) as pool:
        # 使用 map 函数并行处理
        results = pool.map(processfilehelper, args_list)
        for result in results:
            currentinfo, obj_count, img_count, datatype = result
            if datatype not in totalresultdict:
                totalresultdict[datatype] = []
            totalresultdict[datatype].append([currentinfo, obj_count, img_count])
    logger.info('processed totalresultdict: {}'.format(len(totalresultdict)))
    for datatype, resultlist in totalresultdict.items():
        outputfile = osp.join(outputfolder, '%s.json'%(datatype))
        lastobjectcount = 0
        lastimagecount = 0
        images = []
        annotations = []
        for idx, result in enumerate(resultlist):
            currentinfolist, obj_count, img_count = result
            for currentinfo in currentinfolist:
                imageinfo, currentannotations = currentinfo
                imageinfo['id'] += lastimagecount
                imageinfo['file_name'] = imageinfo['file_name'].replace(imageprefix, '')            
                for anno in currentannotations:
                    anno['image_id'] += lastimagecount
                    anno['id'] += lastobjectcount
                    annotations.append(anno)
                images.append(imageinfo)                
                lastobjectcount += obj_count
                lastimagecount += img_count
            train_coco_format_json = dict(
                images = images,
                annotations = annotations,
                categories = ConvertLabelTypeDict
                )
        dump(train_coco_format_json, outputfile)
        logger.info('type:{}, lastimagecount: {}, lastobjectcount : {}'.format(datatype, lastimagecount, lastobjectcount))

def convert_rawpkl_to_coco(inputpklfiletxt, outputfolder):
    datefiledict = {}
    imageprefix = '/data-algorithm-hl/xianglong.meng/data/refine/sample/image/'
    with open(inputpklfiletxt, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            inputpklfile = line.strip()
            image_prefix = os.path.dirname(inputpklfile)
            inputpklbasename = os.path.basename(inputpklfile)
            if 'LJ' not in inputpklfile:
                continue
            datestr = inputpklfile.split('/')[-3]
            if datestr not in datefiledict:
                datefiledict[datestr] = []
            datefiledict[datestr].append(inputpklfile)
            # break
    # logger.info('datefiledict: {}'.format(datefiledict))
    datelist = list(datefiledict.keys())
    traindates = set(datelist[:int(len(datelist)*0.8)])
    valdates = set(datelist[int(len(datelist)*0.8):])
    outputtrainfile = osp.join(outputfolder, 'train.json')
    outputvalfile = osp.join(outputfolder, 'val.json')
    traintotalindex = 0 
    valtotalindex = 0
    trainlabeldict = {'Neg':0}
    vallabeldict = {'Neg':0}
    obj_count = 1
    trainannotations = []
    trainimages = []
    valannotations = []
    valimages = []
    trainsampledict = {}
    valsampledict = {}
    for datestr, inputpklfiles in track_iter_progress(list(datefiledict.items())):
        if datestr in traindates:
            currentindex = traintotalindex
            currentlabeldict = trainlabeldict
            currentannotations = trainannotations
            currentimages = trainimages
            sampledict = trainsampledict
        elif datestr in valdates:
            currentindex = valtotalindex
            currentlabeldict = vallabeldict
            currentannotations = valannotations
            currentimages = valimages
            sampledict = valsampledict
        for inputpklfile in inputpklfiles:
            data_infos = pickle.load(open(inputpklfile, 'rb'))
            logger.info('processing {}'.format(inputpklfile))
            for idx, v in enumerate(data_infos):
                img_path = v['cropimage']
                relativebox = v['relativebox']
                label = v['label']
                localfile = v['localfile']
                file_prefix_name = img_path.replace(imageprefix, '')
                height, width = mmcv.imread(img_path).shape[:2]
                currentimages.append(
                    dict(id = idx + currentindex, file_name = file_prefix_name, height=height, width=width))
                # logger.info('idx: {}, file_prefix_name: {}, height: {}, width: {}'.format(idx, file_prefix_name, height, width))
                if label is not None:
                    if label not in currentlabeldict:
                        currentlabeldict[label] = len(currentlabeldict)
                    if label not in sampledict:
                        sampledict[label] = 0
                    for bbox in relativebox:
                        x_min, y_min, x_max, y_max = bbox
                        data_anno = dict(
                            image_id=idx,
                            id=obj_count,
                            category_id=currentlabeldict[label],
                            bbox=[x_min, y_min, x_max - x_min, y_max - y_min],
                            area=(x_max - x_min) * (y_max - y_min),
                            segmentation=[],
                            iscrowd=0)
                        currentannotations.append(data_anno)
                        obj_count += 1
                        sampledict[label] += 1
                else:
                    data_anno = dict(
                        image_id=idx,
                        id=0,
                        category_id=0,
                        bbox=[],
                        area=0,
                        segmentation=[],
                        iscrowd=0)
                    currentannotations.append(data_anno)
                    if 0 not in sampledict:
                        sampledict[0] = 0
                    sampledict[0] += 1
    traincategories = [{'id':trainlabeldict[key], 'name':key} for key in trainlabeldict]
    valcategories = [{'id':vallabeldict[key], 'name':key} for key in vallabeldict]
    train_coco_format_json = dict(
        images=trainimages,
        annotations=trainannotations,
        categories=traincategories
        )
    dump(train_coco_format_json, outputtrainfile)
    logger.info('trainimages: {}, traincategories : {}, trainsampledict : {}'.format(len(trainimages), traincategories, trainsampledict))
    val_coco_format_json = dict(
        images=valimages,
        annotations=valannotations,
        categories=valcategories
        )
    dump(val_coco_format_json, outputvalfile)
    logger.info('valimages: {}, valcategories {} valsampledict : {}'.format(len(valimages), valcategories, valsampledict))


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
    fmt='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    fire.Fire()
