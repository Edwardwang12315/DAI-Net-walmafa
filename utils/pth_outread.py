# --coding:utf-8--
import torch
import numpy as np


def save_pth_dict_to_txt( pth_path , txt_path ) :
	# 加载 .pth 文件
	data = torch.load( pth_path , map_location = 'cpu' )
	assert isinstance( data , dict ) , "文件内容不是字典！"

	# 递归写入文本
	with open( txt_path , 'w' , encoding = 'utf-8' ) as f :
		def _write_item( key , value , indent ) :
			# 处理张量
			if isinstance( value , torch.Tensor ) :
				np_value = value.cpu().numpy()
				value_str = np.array2string( np_value , precision = 4 , separator = ', ' , suppress_small = True )
				f.write( ' ' * indent + f"{key} (Tensor {value.shape}):\n{' ' * (indent + 2)}{value_str}\n" )
			# 处理嵌套字典
			elif isinstance( value , dict ) :
				f.write( ' ' * indent + f"{key}:\n" )
				for sub_key , sub_value in value.items() :
					_write_item( sub_key , sub_value , indent + 4 )
			# 其他类型
			else :
				f.write( ' ' * indent + f"{key}: {value}\n" )

		for key , value in data.items() :
			_write_item( key , value , indent = 0 )

save_pth_dict_to_txt( 'tar_mod.pth' , 'tar_mod.txt' )
