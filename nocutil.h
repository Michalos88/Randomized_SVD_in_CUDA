
#ifndef _NOCUTIL_H_
#define _NOCUTIL_H_

// includes, file
#include "nocutil.h"
#include <string>

// includes, system
#include <fstream>
#include <vector>
#include <iostream>
//#include <algorithm>
//#include <math.h>




//////////////////////////////////////////////////////////////////////////////
    //! Read file \filename and return the data
    //! @return CUTTrue if reading the file succeeded, otherwise false
    //! @param filename name of the source file
    //! @param data  uninitialized pointer, returned initialized and pointing to
    //!        the data read
    //! @param len  number of data elements in data, -1 on error
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    bool
    nocutReadFile( const char* filename, T** data, unsigned int* len, bool verbose) 
    {
        // check input arguments
        if( NULL == filename || NULL == len)
		{
			exit(2);
		}

        // intermediate storage for the data read
        std::vector<T>  data_read;

        // open file for reading
        std::fstream fh( filename, std::fstream::in);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutReadFile() : Opening file failed." << std::endl;
            return false;
        }

        // read all data elements 
        T token;
        while( fh.good()) 
        {
            fh >> token;   
            data_read.push_back( token);
        }

        // the last element is read twice
        data_read.pop_back();

        // check if reading result is consistent
        if( ! fh.eof()) 
        {
            if (verbose)
                std::cerr << "WARNING : readData() : reading file might have failed." 
                << std::endl;
        }

        fh.close();

        // check if the given handle is already initialized
        if( NULL != *data) 
        {
            if( *len != data_read.size()) 
            {
                std::cerr << "cutReadFile() : Initialized memory given but "
                          << "size  mismatch with signal read "
                          << "(data read / data init = " << (unsigned int)data_read.size()
                          <<  " / " << *len << ")" << std::endl;

                return false;
            }
        }
        else 
        {
            // allocate storage for the data read
			*data = (T*) malloc( sizeof(T) * data_read.size());
            // store signal size
            *len = static_cast<unsigned int>( data_read.size());
        }

        // copy data
        memcpy( *data, &data_read.front(), sizeof(T) * data_read.size());

        return true;
    }


////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
bool
nocutReadFilei( const char* filename, int** data, unsigned int* len, bool verbose) 
{
    return nocutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg unsigned integer data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
bool
nocutReadFileui( const char* filename, unsigned int** data, unsigned int* len, bool verbose) 
{
    return nocutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////////
//! Read file \filename containg single precision floating point data 
//! @return CUTTrue if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
////////////////////////////////////////////////////////////////////////////////
bool
nocutReadFilef( const char* filename, float** data, unsigned int* len, bool verbose) 
{
    return nocutReadFile( filename, data, len, verbose);
}

////////////////////////////////////////////////////////////////////////////// 
    //! Compare two arrays of arbitrary type       
    //! @return  true if \a reference and \a data are identical, otherwise false
    //! @param reference  handle to the reference data / gold image
    //! @param data       handle to the computed data
    //! @param len        number of elements in reference and data
    //! @param epsilon    epsilon to use for the comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T, class S>
    bool  
    nocompareData( const T* reference, const T* data, const unsigned int len, 
                 const S epsilon, const float threshold) 
    {
        if( epsilon < 0)
			exit(3);

        bool result = true;
        unsigned int error_count = 0;

        for( unsigned int i = 0; i < len; ++i) {

            T diff = reference[i] - data[i];
            bool comp = (diff <= epsilon) && (diff >= -epsilon);
            result &= comp;

            error_count += !comp;

#ifdef _DEBUG
            if( ! comp) 
            {
                std::cerr << "ERROR, i = " << i << ",\t " 
                    << reference[i] << " / "
                    << data[i] 
                    << " (reference / data)\n";
            }
#endif
        }

        if (threshold == 0.0f) {
            return (result) ? true : false;
        } else {
            if (error_count) {
                printf("%4.2f(%%) of bytes mismatched (count=%d)\n", (float)error_count*100/(float)len, error_count);
            }

            return (len*threshold > error_count) ? true : false;
        }
    }



bool
nocutComparefe( const float* reference, const float* data,
             const unsigned int len, const float epsilon ) 
{
    return nocompareData( reference, data, len, epsilon, 0.0f);
}

//////////////////////////////////////////////////////////////////////////////
    //! Write a data file \filename 
    //! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
    //! @param filename name of the source file
    //! @param data  data to write
    //! @param len  number of data elements in data, -1 on error
    //! @param epsilon  epsilon for comparison
    //////////////////////////////////////////////////////////////////////////////
    template<class T>
    bool
    nocutWriteFile( const char* filename, const T* data, unsigned int len,
                  const T epsilon, bool verbose) 
    {
        if( NULL == filename || NULL == data)
			exit(2);

        // open file for writing
        std::fstream fh( filename, std::fstream::out);
        // check if filestream is valid
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Opening file failed." << std::endl;
            return false;
        }

        // first write epsilon
        fh << "# " << epsilon << "\n";

        // write data
        for( unsigned int i = 0; (i < len) && (fh.good()); ++i) 
        {
            fh << data[i] << ' ';
        }

        // Check if writing succeeded
        if( ! fh.good()) 
        {
            if (verbose)
                std::cerr << "cutWriteFile() : Writing file failed." << std::endl;
            return false;
        }

        // file ends with nl
        fh << std::endl;

        return true;
    }


////////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename for single precision floating point data
//! @return CUTTrue if writing the file succeeded, otherwise CUTFalse
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
////////////////////////////////////////////////////////////////////////////////
bool
nocutWriteFilef( const char* filename, const float* data, unsigned int len,
               const float epsilon, bool verbose=false) 
{
    return nocutWriteFile( filename, data, len, epsilon, verbose);
}




#endif