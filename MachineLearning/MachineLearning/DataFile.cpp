#include "DataFile.h"

namespace Utils
{

	DataFileWriter::DataFileWriter(const std::string& name)
		: file(name, std::ios::out | std::ios::trunc)
	{
	}

}
