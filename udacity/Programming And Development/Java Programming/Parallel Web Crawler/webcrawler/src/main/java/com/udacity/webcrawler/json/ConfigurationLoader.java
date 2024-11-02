package com.udacity.webcrawler.json;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;
import com.fasterxml.jackson.core.exc.StreamReadException;
import com.fasterxml.jackson.databind.DatabindException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.annotation.JsonDeserialize;

/**
 * A static utility class that loads a JSON configuration file.
 */
@JsonDeserialize(builder = CrawlerConfiguration.Builder.class)
public final class ConfigurationLoader {

  private final Path path;
  static final ObjectMapper objm = new ObjectMapper();


  /**
   * Create a {@link ConfigurationLoader} that loads configuration from the given {@link Path}.
   */
  public ConfigurationLoader(Path path) {
    this.path = Objects.requireNonNull(path);
  }

  /**
   * Loads configuration from this {@link ConfigurationLoader}'s path
   *
   * @return the loaded {@link CrawlerConfiguration}.
   * @throws IOException
   */

  public CrawlerConfiguration load() {
    // Fill in this method.
    try(BufferedReader br = Files.newBufferedReader(path)) {
      return read(br);
    } catch (Exception e) {
      e.printStackTrace();
      return null;
    }
  }

  /**
   * Loads crawler configuration from the given reader.
   *
   * @param reader a Reader pointing to a JSON string that contains crawler configuration.
   * @return a crawler configuration
   * @throws IOException
   * @throws DatabindException
   * @throws StreamReadException
   */
  public static CrawlerConfiguration read(Reader reader) {
    // This is here to get rid of the unused variable warning.
    Objects.requireNonNull(reader);
    // Fill in this method
    objm.disable(com.fasterxml.jackson.core.JsonParser.Feature.AUTO_CLOSE_SOURCE);
    try {
      return objm.readValue(reader, CrawlerConfiguration.Builder.class).build();
    } catch (Exception e) {
      // handle exception
      e.printStackTrace();
      return null;
    }
  }
}
