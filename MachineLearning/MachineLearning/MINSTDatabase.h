#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "DataFileBase.h"

namespace Utils {

	class MINSTDatabase
	{
	public:
		MINSTDatabase() {}

		virtual ~MINSTDatabase() 
		{
			Close();
		}
		
		void setRelativePath(const std::string& p)
		{
			imagesFile.setRelativePath(p);
			labelsFile.setRelativePath(p);
		}
		
		void setImagesFileName(const std::string& d)
		{
			imagesFile.setDataFileName(d);
		}
		
		void setLabelsFileName(const std::string& d)
		{
			labelsFile.setDataFileName(d);
		}
		
		std::string getImagesFilePath() const
		{
			return imagesFile.getFilePath();
		}
		
		std::string getLabelsFilePath() const
		{
			return labelsFile.getFilePath();
		}

		bool Open()
		{
			imagesStream.open(imagesFile.getFilePath(), std::ios::binary);
			if (!imagesStream.good()) return false;
			// check magic
			uint32_t magic = 0;
			imagesStream.read((char*)&magic, sizeof(uint32_t));
			if (magic != 0x803) 
			{
				imagesStream.close();
				return false;
			}

			uint32_t cntImgs = 0;
			imagesStream.read((char*)&cntImgs, sizeof(uint32_t));

			uint32_t rows = 0;	
			imagesStream.read((char*)&rows, sizeof(uint32_t));
			uint32_t cols = 0;
			imagesStream.read((char*)&cols, sizeof(uint32_t));

			if (rows != cols || rows != 28)
			{
				imagesStream.close();
				return false;
			}

			labelsStream.open(labelsFile.getFilePath(), std::ios::binary);
			if (!labelsStream.good()) 
			{
				imagesStream.close();
				return false;
			}
			// check magic
			magic = 0;
			labelsStream.read((char*) magic, sizeof(uint32_t));
			if (magic != 0x801) 
			{
				Close();

				return false;
			}

			uint32_t cntLabels = 0;
			labelsStream.read((char*)&cntLabels, sizeof(uint32_t));

			if (cntLabels != cntImgs)
			{
				Close();

				return false;
			}

			return true;
		}

		void Close()
		{
			imagesStream.close();
			labelsStream.close();
		}

		bool ReadImage(std::vector<uint8_t>& img)
		{
			if (!imagesStream.good()) return false;
			img.resize(28 * 28);
			imagesStream.read((char*)img.data(), img.size());
			return true;
		}

		bool ReadLabel(uint8_t& label)
		{
			if (!labelsStream.good()) return false;
			labelsStream.read((char*)&label, sizeof(uint8_t));
			return true;
		}


		std::pair<std::vector<uint8_t>, uint8_t> ReadImageAndLabel()
		{
			std::vector<uint8_t> img;
			uint8_t label = 0xFF; // invalid, read values will be 0-9

			if (ReadImage(img))
				ReadLabel(label);
			
			return std::make_pair(img, label);
		}

		std::vector<std::pair<std::vector<uint8_t>, uint8_t>> ReadAllImagesAndLabels()
		{
			std::vector<std::pair<std::vector<uint8_t>, uint8_t>> res;

			while (imagesStream.good() && labelsStream.good())
			{
				auto imgAndLabel = ReadImageAndLabel();
				if (imgAndLabel.second != 0xFF) res.emplace_back(imgAndLabel);
				else break;
			}

			return res;
		}

	protected:
		DataFileBase imagesFile;
		DataFileBase labelsFile;

		std::ifstream imagesStream;
		std::ifstream labelsStream;
	};

}


