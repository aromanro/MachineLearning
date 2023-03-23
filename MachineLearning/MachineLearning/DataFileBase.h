#pragma once

#include <string>
#include <filesystem>

namespace Utils {

	class DataFileBase
	{
	public:
		DataFileBase() {}
		virtual ~DataFileBase() {}

		std::string getFilePath() const
		{
			std::filesystem::path p = std::filesystem::current_path();
			if (relPath[0] != '/' && relPath[0] != '\\')
				p += '/';

			p += relPath;
			p += dataFileName;

			return p.string();
		}

		void setRelativePath(const std::string& p)
		{
			relPath = p;
			if (relPath.empty()) return;

			if (relPath[relPath.length() - 1] != '/' && relPath[relPath.length() - 1] != '\\')
				relPath += "/";
		}

		void setDataFileName(const std::string& d)
		{
			dataFileName = d;
		}


	protected:
		std::string relPath = "../../data/";
		std::string dataFileName = "data.txt";
	};

}


