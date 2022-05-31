from sys import stdout 
from loguru import logger 


log_format = [
	'<W><k>{time:YYYY-MM-DD hh:mm:ss}</k></W>',
	'<c>{file:^15}</c>',
	'<e>{line:03d}</e>',
	'<r>{level:^10}</r>',
	'<Y><k>{message:<50}</k></Y>'
]

log_separator = ' | '

logger.remove()
logger.add(
	sink=stdout, 
	level='TRACE', 
	format=log_separator.join(log_format)
)

if __name__ == '__main__':
	logger.debug('check if log is available...!')